"""STEM pre-training.

Single GPU:  python pretrain.py --data_dir <processed_data_root> --save_dir <output_dir>
Multi GPU :  torchrun --nproc_per_node=4 pretrain.py --data_dir <processed_data_root> --save_dir <output_dir>
"""
import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from datasets import PretrainingDataset, seed_worker
from model import STEM
from utils import PRETRAIN_DATASETS, load_state_dict, makedirs, setup_seed


def get_args():
    parser = argparse.ArgumentParser('STEM pre-training')
    parser.add_argument('--data_dir', type=str, required=True, help='root directory of the preprocessed datasets')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save checkpoints')
    parser.add_argument('--datasets', type=str, nargs='+', default=PRETRAIN_DATASETS)
    parser.add_argument('--resume', type=str, default='', help='checkpoint (latest.pth) to resume from')
    parser.add_argument('--seed', type=int, default=42)

    # optimization
    parser.add_argument('--batch_size', type=int, default=8, help='number of episodes per GPU')
    parser.add_argument('--max_iters', type=int, default=3000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=50000)

    # STEM
    parser.add_argument('--n_sup', type=int, default=4, help='trials per class in an episode (query + K supports)')
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--patch_size', type=int, default=40)
    parser.add_argument('--seq_len', type=int, default=20, help='number of patches per sample')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=8)
    return parser.parse_args()


def mean_excluding_self(x):
    """(B, N, F) -> leave-one-out mean over N."""
    return (x.sum(dim=1, keepdim=True) - x) / (x.shape[1] - 1)


def info_nce_loss(z1, z2, temperature):
    """SimCLR NT-Xent loss between two views."""
    n = z1.shape[0]
    features = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    similarity = features @ features.T
    similarity = similarity.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z1.device), float('-inf'))
    targets = (torch.arange(2 * n, device=z1.device) + n) % (2 * n)
    return F.cross_entropy(similarity / temperature, targets)


class Trainer:
    def __init__(self, args, dataset, model, device, distributed):
        self.args = args
        self.device = device
        self.distributed = distributed
        self.is_main = not distributed or dist.get_rank() == 0

        model = model.to(device)
        self.model = DDP(model, device_ids=[device.index], find_unused_parameters=True,
                         broadcast_buffers=False) if distributed else model
        self.data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True, collate_fn=dataset.collate, worker_init_fn=seed_worker,
                                      persistent_workers=args.num_workers > 0)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.max_iters,
                                                                    eta_min=args.min_lr)
        self.iteration = 0
        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu')
            self.unwrap().load_state_dict(load_state_dict(args.resume), strict=False)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.iteration = ckpt['iteration']
            print(f'Resumed from {args.resume} (iteration {self.iteration})')

    def unwrap(self):
        return self.model.module if self.distributed else self.model

    def forward(self, x, ch, padding_mask):
        """Two passes over the same channel subset: a masked view for L_self and a clean view for all losses."""
        bz, ch_num, patch_num, _ = x.shape
        n_drop = random.randint(1, ch_num // 2)
        keep = torch.tensor(sorted(random.sample(range(ch_num), ch_num - n_drop)), device=x.device)
        x, ch, padding_mask = x[:, keep], ch[:, keep], 1 - padding_mask[:, keep]

        mask = torch.zeros((bz, len(keep), patch_num), dtype=torch.long, device=x.device).bernoulli_(
            self.args.mask_ratio)
        z_self_masked, _, _ = self.model(x, ch, mask=mask, padding_mask=padding_mask)
        z_self, z_task, z_sub = self.model(x, ch, padding_mask=padding_mask)
        return z_self_masked, z_self, z_task, z_sub

    def compute_loss(self, batch):
        args = self.args
        s = args.n_sup
        x, z, n, xp, label, mx, mz, mn, mxp, chx, chz, chn, chxp = [b.to(self.device, non_blocking=True)
                                                                    for b in batch]
        bz = x.shape[0]
        # (B, C, T, S) -> (B*S, C, P, L)
        x = x.permute(0, 3, 1, 2).reshape(bz * s, -1, args.seq_len, args.patch_size)
        z = z.permute(0, 3, 1, 2).reshape(bz * s, -1, args.seq_len, args.patch_size)
        n = n.reshape(bz, -1, args.seq_len, args.patch_size)
        xp = xp.reshape(bz, -1, args.seq_len, args.patch_size)
        chx = chx.permute(0, 3, 1, 2).reshape(bz * s, -1, 3)
        chz = chz.permute(0, 3, 1, 2).reshape(bz * s, -1, 3)
        mx = mx.unsqueeze(1).expand(-1, s, -1).reshape(bz * s, -1)
        mz = mz.unsqueeze(1).expand(-1, s, -1).reshape(bz * s, -1)

        z_self_masked, z_self, tf, sf = self.forward(torch.cat([x, z, n, xp]), torch.cat([chx, chz, chn, chxp]),
                                                     torch.cat([mx, mz, mn, mxp]))
        t = args.temperature

        # L_self: contrastive learning between the masked and the clean view
        ssl_loss = info_nce_loss(F.normalize(z_self, dim=-1), F.normalize(z_self_masked, dim=-1), t)

        tf = F.normalize(tf, dim=-1)
        sf = F.normalize(sf, dim=-1)
        tf_x, tf_z, _, _ = torch.split(tf, [bz * s, bz * s, bz, bz])
        sf_x, sf_z, sf_n, sf_xp = torch.split(sf, [bz * s, bz * s, bz, bz])
        tf_x, tf_z = tf_x.reshape(bz, s, -1), tf_z.reshape(bz, s, -1)
        sf_x, sf_z = sf_x.reshape(bz, s, -1), sf_z.reshape(bz, s, -1)

        # L_task: 2-way K-shot, positive prototype = same class, negative prototype = other class (same subject)
        tf_x_mean, tf_z_mean = mean_excluding_self(tf_x), mean_excluding_self(tf_z)
        query = torch.cat([tf_x, tf_z], dim=1).reshape(-1, tf.shape[-1])
        pos = F.cosine_similarity(query, torch.cat([tf_x_mean, tf_z_mean], dim=1).reshape(-1, tf.shape[-1]))
        neg = F.cosine_similarity(query, torch.cat([tf_z_mean, tf_x_mean], dim=1).reshape(-1, tf.shape[-1]))
        task_weight = label.float().expand(-1, 2 * s).reshape(-1)  # only episodes with valid task labels
        task_loss = F.cross_entropy(torch.stack([pos, neg], dim=-1) / t, torch.zeros_like(pos, dtype=torch.long),
                                    reduction='none')
        task_loss = (task_loss * task_weight).sum() / (task_weight.sum() + 1e-6)

        # L_sub: positive prototype = same subject (other class), negative = single trial of another subject
        sf_x_mean, sf_z_mean = mean_excluding_self(sf_x), mean_excluding_self(sf_z)
        query = torch.cat([sf_x, sf_z], dim=1).reshape(-1, sf.shape[-1])
        pos = F.cosine_similarity(query, torch.cat([sf_z_mean, sf_x_mean], dim=1).reshape(-1, sf.shape[-1]))
        neg = F.cosine_similarity(query, torch.cat([sf_xp.unsqueeze(1).expand(-1, s, -1),
                                                    sf_n.unsqueeze(1).expand(-1, s, -1)], dim=1).reshape(-1, sf.shape[-1]))
        subject_loss = F.cross_entropy(torch.stack([pos, neg], dim=-1) / t, torch.zeros_like(pos, dtype=torch.long))

        return task_loss + subject_loss + ssl_loss, task_loss, subject_loss, ssl_loss

    def save(self, name, full=False):
        state = self.unwrap().state_dict()
        if full:
            state = {'model': state, 'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict(), 'iteration': self.iteration}
        torch.save(state, os.path.join(self.args.save_dir, name))

    def train(self):
        self.model.train()
        while self.iteration < self.args.max_iters:
            for batch in self.data_loader:
                loss, task_loss, subject_loss, ssl_loss = self.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
                self.optimizer.step()
                self.scheduler.step()
                self.iteration += 1

                if self.is_main and self.iteration % self.args.log_every == 0:
                    print(f'Iter {self.iteration}/{self.args.max_iters} | loss {loss.item():.4f} | '
                          f'task {task_loss.item():.4f} | subject {subject_loss.item():.4f} | '
                          f'self {ssl_loss.item():.4f} | lr {self.scheduler.get_last_lr()[0]:.2e}', flush=True)
                if self.is_main and (self.iteration % self.args.save_every == 0
                                     or self.iteration == self.args.max_iters):
                    self.save(f'stem_iter{self.iteration}.pth')
                    self.save('latest.pth', full=True)
                if self.iteration >= self.args.max_iters:
                    break


def main():
    args = get_args()
    distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # episodes are sampled randomly inside the dataset, so every rank needs a different seed
    setup_seed(args.seed + rank)
    if rank == 0:
        makedirs(args.save_dir)
        print(args)

    dataset = PretrainingDataset(args.data_dir, args.datasets, n_sup=args.n_sup)
    model = STEM(args.patch_size, args.patch_size, args.d_model, args.dim_feedforward, args.n_layer, args.nhead)
    Trainer(args, dataset, model, device, distributed).train()

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
