"""Evaluates a fine-tuned STEM model on the test subjects of a downstream dataset.

python evaluate.py --data_dir <processed_data_root> --dataset PhysioNet --ckpt <stem_PhysioNet.pth>
"""
import argparse

from torch.utils.data import DataLoader

from datasets import DownstreamDataset
from finetune import evaluate, setup
from utils import DOWNSTREAM_DATASETS, format_metrics, load_state_dict


def main():
    parser = argparse.ArgumentParser('STEM evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='root directory of the preprocessed datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DOWNSTREAM_DATASETS))
    parser.add_argument('--ckpt', type=str, required=True, help='fine-tuned STEM weights')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patch_size', type=int, default=40)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=8)
    args = parser.parse_args()

    model = setup(args)
    model.load_state_dict(load_state_dict(args.ckpt))
    model.to(args.device)

    dataset = DownstreamDataset(args.data_dir, args.dataset, args.split, args.seq_len, args.patch_size, args.num_cls)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate(model, loader, args)
    print(f'{args.dataset} {args.split} | {format_metrics(metrics)}')
    print(metrics['cm'])


if __name__ == '__main__':
    main()
