"""Fine-tuning STEM on a downstream dataset (cross-subject).

python finetune.py --data_dir <processed_data_root> --dataset PhysioNet --pretrained <stem_pretrained.pth> --save_dir <output_dir>
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from datasets import DownstreamDataset, seed_worker
from model import STEM
from utils import (DOWNSTREAM_DATASETS, compute_metrics, format_metrics, load_pretrained, makedirs,
                   selection_score, setup_seed)

HEAD_MODULES = ['classifier', 'decomposer', 'encoder_task', 'encoder_sub', 'decoder', 'proj_task', 'proj_subject',
                'proj_out']


def get_parser():
    parser = argparse.ArgumentParser('STEM fine-tuning')
    parser.add_argument('--data_dir', type=str, required=True, help='root directory of the preprocessed datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DOWNSTREAM_DATASETS))
    parser.add_argument('--pretrained', type=str, default='', help='pre-trained STEM weights')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save the fine-tuned model')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    # optimization
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate of the patch embedding and encoder')
    parser.add_argument('--head_lr_scale', type=float, default=10.0,
                        help='lr multiplier for decomposer / decoders / classifier')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--loss_weight_ce', type=float, default=1.0)
    parser.add_argument('--loss_weight_task', type=float, default=1.0)
    parser.add_argument('--loss_weight_subject', type=float, default=1.0)

    # STEM
    parser.add_argument('--patch_size', type=int, default=40)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=8)
    return parser


def setup(args):
    """Adds dataset specific settings to args and builds the model."""
    cfg = DOWNSTREAM_DATASETS[args.dataset]
    args.num_cls, args.num_ch = cfg['num_cls'], cfg['num_ch']
    args.seq_len = cfg['length'] // args.patch_size
    args.binary = args.num_cls == 2
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    return STEM(args.patch_size, args.patch_size, args.d_model, args.dim_feedforward, args.n_layer, args.nhead,
                num_cls=args.num_cls, num_ch=args.num_ch)


def build_loader(dataset, args, train):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers,
                      worker_init_fn=seed_worker, pin_memory=True)


def build_optimizer(model, args):
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        (head_params if any(m in name for m in HEAD_MODULES) else backbone_params).append(param)
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': args.lr},
                                   {'params': head_params, 'lr': args.lr * args.head_lr_scale}],
                                  weight_decay=args.weight_decay)
    # linear warm-up followed by cosine annealing, stepped once per epoch
    warmup = min(args.warmup_epochs, max(args.epochs - 1, 0))
    scheduler = SequentialLR(optimizer, milestones=[warmup], schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(warmup, 1)),
        CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup, 1), eta_min=args.min_lr)])
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, args):
    model.eval()
    truths, scores = [], []
    for x, y, ch in loader:
        pred, _, _ = model(x.to(args.device), ch.to(args.device))
        scores.append(pred.softmax(dim=-1).cpu())
        truths.append(y)
    return compute_metrics(torch.cat(truths).numpy(), torch.cat(scores).numpy(), args.binary)


def meta_losses(model, tf, sf, z, ch_z, p, ch_p, t):
    """L_task and L_sub on a downstream batch.

    z: trials of the same subject with the other classes (task negatives, subject positives)
    p: trial of a different subject with the same class (task positive, subject negative)
    """
    b, n_other = z.shape[:2]
    _, tf_z, sf_z = model(z.flatten(0, 1), ch_z.flatten(0, 1))
    _, tf_p, sf_p = model(p, ch_p)
    tf, tf_z, tf_p = F.normalize(tf, dim=-1), F.normalize(tf_z, dim=-1).reshape(b, n_other, -1), F.normalize(tf_p, dim=-1)
    sf, sf_z, sf_p = F.normalize(sf, dim=-1), F.normalize(sf_z, dim=-1).reshape(b, n_other, -1), F.normalize(sf_p, dim=-1)
    target = torch.zeros(b, dtype=torch.long, device=tf.device)

    task_logits = [F.cosine_similarity(tf, tf_p)] + [F.cosine_similarity(tf, tf_z[:, i]) for i in range(n_other)]
    task_loss = F.cross_entropy(torch.stack(task_logits, dim=-1) / t, target)

    sf_neg = F.cosine_similarity(sf, sf_p)
    subject_loss = torch.stack([
        F.cross_entropy(torch.stack([F.cosine_similarity(sf, sf_z[:, i]), sf_neg], dim=-1) / t, target)
        for i in range(n_other)]).mean()
    return task_loss, subject_loss


def train(model, args):
    loaders = {split: build_loader(DownstreamDataset(args.data_dir, args.dataset, split, args.seq_len,
                                                     args.patch_size, args.num_cls), args, split == 'train')
               for split in ['train', 'val', 'test']}
    optimizer, scheduler = build_optimizer(model, args)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    device = args.device

    best_score, best_epoch, best_state = -np.inf, 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for it, (x, y, ch, z, ch_z, p, ch_p) in enumerate(loaders['train']):
            x, y, ch = x.to(device), y.to(device), ch.to(device)
            pred, tf, sf = model(x, ch)
            ce_loss = criterion(pred, y)
            task_loss, subject_loss = meta_losses(model, tf, sf, z.to(device), ch_z.to(device), p.to(device),
                                                  ch_p.to(device), args.temperature)
            loss = args.loss_weight_ce * ce_loss + args.loss_weight_task * task_loss \
                + args.loss_weight_subject * subject_loss

            optimizer.zero_grad()
            loss.backward()
            if args.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()

            total += loss.item() * x.shape[0]
            count += x.shape[0]
            print(f'Epoch {epoch} Iter {it + 1}/{len(loaders["train"])} | loss {loss.item():.4f} | '
                  f'ce {ce_loss.item():.4f} | task {task_loss.item():.4f} | subject {subject_loss.item():.4f}',
                  end='\r')
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        metrics = evaluate(model, loaders['val'], args)
        print(f'\nEpoch {epoch}: train loss {total / count:.5f}, lr {lr:.2e} | val {format_metrics(metrics)}')
        if selection_score(metrics) > best_score:
            best_score, best_epoch = selection_score(metrics), epoch
            best_state = copy.deepcopy(model.state_dict())
            print('  -> best model on validation set')

    model.load_state_dict(best_state)
    metrics = evaluate(model, loaders['test'], args)
    print(f'Best epoch {best_epoch} | test {format_metrics(metrics)}')
    print(metrics['cm'])

    makedirs(args.save_dir)
    torch.save(best_state, os.path.join(args.save_dir, f'stem_{args.dataset}.pth'))
    with open(os.path.join(args.save_dir, f'stem_{args.dataset}_test.json'), 'w') as f:
        json.dump({'best_epoch': best_epoch, **{k: float(v) for k, v in metrics.items() if k != 'cm'},
                   'confusion_matrix': metrics['cm'].tolist(), 'args': vars(args)}, f, indent=2)
    print(f'Saved to {args.save_dir}')


def main():
    args = get_parser().parse_args()
    setup_seed(args.seed)
    model = setup(args)
    if args.pretrained:
        load_pretrained(model, args.pretrained)
    else:
        print('No pre-trained weights are given: training from scratch.')
    print(args)
    train(model.to(args.device), args)


if __name__ == '__main__':
    main()
