"""Few-shot adaptation of STEM to unseen target subjects (Fig. 2 of the paper).

The pre-trained model is fine-tuned on the training subjects with L_task and L_sub using K' support trials per class.
Test queries are classified by the cosine similarity between z_task of the query and the class prototypes computed from
K' labeled trials of the same (unseen) subject, after a few adaptation steps on the support set.

python fewshot.py --data_dir <processed_data_root> --dataset PhysioNet --few 5 --pretrained <stem_pretrained.pth> --save_dir <output_dir>
"""
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import FewShotDataset, seed_worker
from finetune import build_optimizer, get_parser, setup
from utils import compute_metrics, format_metrics, load_pretrained, load_state_dict, makedirs, selection_score, \
    setup_seed


def get_args():
    parser = get_parser()
    parser.description = 'STEM few-shot adaptation'
    parser.add_argument('--few', type=int, default=5, help="number of support trials per class (K')")
    parser.add_argument('--adapt_steps', type=int, default=10, help='adaptation steps on the support set at test time')
    parser.add_argument('--adapt_lr', type=float, default=1e-4)
    parser.add_argument('--ckpt', type=str, default='', help='few-shot fine-tuned weights (skips training)')
    parser.set_defaults(epochs=30, batch_size=16, head_lr_scale=1.0)
    return parser.parse_args()


def prototypes(model, z, ch_z):
    """z: (B, N, K', C, P, L) -> class prototypes (B, N, F) and normalized support features (B, N, K', F)."""
    b, n, k = z.shape[:3]
    _, tf, sf = model(z.flatten(0, 2), ch_z.flatten(0, 2))
    tf = F.normalize(tf, dim=-1).reshape(b, n, k, -1)
    sf = F.normalize(sf, dim=-1).reshape(b, n, k, -1)
    return tf, sf


def adapt(model, z, ch_z, args):
    """Leave-one-out prototypical adaptation on the support set of the target subject."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.adapt_lr, weight_decay=args.weight_decay)
    for _ in range(args.adapt_steps):
        tf, _ = prototypes(model, z, ch_z)
        tf = tf.permute(0, 2, 1, 3)  # (B, K', N, F)
        proto = (tf.sum(dim=1, keepdim=True) - tf) / (args.few - 1)  # prototypes without the k-th support trial
        logits = F.cosine_similarity(tf.unsqueeze(3), proto.unsqueeze(2), dim=-1)  # (B, K', N, N)
        target = torch.arange(args.num_cls, device=z.device).repeat(tf.shape[0] * args.few)
        loss = F.cross_entropy(logits.reshape(-1, args.num_cls) / args.temperature, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate(model, loader, args, adapt_steps=0):
    device = args.device
    truths, scores = [], []
    state_dict = copy.deepcopy(model.state_dict()) if adapt_steps > 0 else None
    for i, (x, y, ch, z, ch_z) in enumerate(loader):
        x, ch, z, ch_z = x.to(device), ch.to(device), z.to(device), ch_z.to(device)
        if adapt_steps > 0:
            model.train()
            adapt(model, z, ch_z, args)
            print(f'Adapting {i + 1}/{len(loader)}', end='\r')
        model.eval()
        with torch.no_grad():
            _, tf_q, _ = model(x, ch)
            tf_s, _ = prototypes(model, z, ch_z)
            scores.append(F.cosine_similarity(F.normalize(tf_q, dim=-1).unsqueeze(1), tf_s.mean(dim=2), dim=-1).cpu())
        truths.append(y)
        if adapt_steps > 0:
            model.load_state_dict(state_dict)
    return compute_metrics(torch.cat(truths).numpy(), torch.cat(scores).numpy(), args.binary)


def train(model, loaders, args):
    optimizer, scheduler = build_optimizer(model, args)
    t = args.temperature
    best_score, best_epoch, best_state = -np.inf, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for it, (x, y, ch, z, ch_z) in enumerate(loaders['train']):
            x, y, ch, z, ch_z = [v.to(args.device) for v in (x, y, ch, z, ch_z)]
            _, tf_q, sf_q = model(x, ch)
            tf_q, sf_q = F.normalize(tf_q, dim=-1), F.normalize(sf_q, dim=-1)
            tf_s, sf_s = prototypes(model, z, ch_z)
            tf_s, sf_s = tf_s.mean(dim=2), sf_s.mean(dim=2)  # (B, N, F)

            # L_task: prototypes of the same subject and of another subject in the batch
            logits_same = F.cosine_similarity(tf_q.unsqueeze(1), tf_s, dim=-1) / t
            logits_other = F.cosine_similarity(tf_q.unsqueeze(1), tf_s[torch.randperm(x.shape[0])], dim=-1) / t
            task_loss = 0.5 * F.cross_entropy(logits_same, y) + 0.5 * F.cross_entropy(logits_other, y)

            # L_sub: for each class, the query has to match the prototype of its own subject within the batch
            sf_sim = F.cosine_similarity(sf_q[None, None], sf_s.permute(1, 0, 2).unsqueeze(2), dim=-1) / t  # (N, B, B)
            subject_loss = torch.stack([
                F.cross_entropy(sf_sim[:, :, i], torch.full((sf_sim.shape[0],), i, device=x.device))
                for i in range(sf_sim.shape[-1])]).mean()

            loss = task_loss + subject_loss
            optimizer.zero_grad()
            loss.backward()
            if args.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()

            total += loss.item() * x.shape[0]
            count += x.shape[0]
            print(f'Epoch {epoch} Iter {it + 1}/{len(loaders["train"])} | loss {loss.item():.4f} | '
                  f'task {task_loss.item():.4f} | subject {subject_loss.item():.4f}', end='\r')
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        metrics = evaluate(model, loaders['val'], args)
        print(f'\nEpoch {epoch}: train loss {total / count:.5f}, lr {lr:.2e} | val {format_metrics(metrics)}')
        if selection_score(metrics) > best_score:
            best_score, best_epoch = selection_score(metrics), epoch
            best_state = copy.deepcopy(model.state_dict())
            print('  -> best model on validation set')

    model.load_state_dict(best_state)
    makedirs(args.save_dir)
    torch.save(best_state, os.path.join(args.save_dir, f'stem_{args.dataset}_{args.few}shot.pth'))
    return best_epoch


def main():
    args = get_args()
    setup_seed(args.seed)
    model = setup(args)
    if args.few < 2 and args.adapt_steps > 0:
        print("Leave-one-out adaptation needs K' >= 2: disabling test-time adaptation.")
        args.adapt_steps = 0
    print(args)

    def loader(split, batch_size):
        dataset = FewShotDataset(args.data_dir, args.dataset, split, args.seq_len, args.patch_size, args.num_cls,
                                 args.few)
        return DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train', num_workers=args.num_workers,
                          worker_init_fn=seed_worker, pin_memory=True)

    best_epoch = None
    if args.ckpt:
        model.load_state_dict(load_state_dict(args.ckpt))
        model.to(args.device)
    else:
        if args.pretrained:
            load_pretrained(model, args.pretrained)
        model.to(args.device)
        loaders = {'train': loader('train', args.batch_size), 'val': loader('val', args.batch_size)}
        best_epoch = train(model, loaders, args)

    # each test query is adapted independently with its own support set
    metrics = evaluate(model, loader('test', 1), args, adapt_steps=args.adapt_steps)
    print(f'\n{args.few}-shot test | {format_metrics(metrics)}')
    print(metrics['cm'])
    if args.save_dir:
        makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, f'stem_{args.dataset}_{args.few}shot_test.json'), 'w') as f:
            json.dump({'best_epoch': best_epoch, **{k: float(v) for k, v in metrics.items() if k != 'cm'},
                       'confusion_matrix': metrics['cm'].tolist(), 'args': vars(args)}, f, indent=2)


if __name__ == '__main__':
    main()
