"""SEED-FRA / SEED-GER emotion datasets (pretraining, used without labels).

Writes flat files to <out_dir>/<SEED-FRA|SEED-GER>/*.pkl; run regroup_subjects.py afterwards.
"""
import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mne

from common import L_FREQ, H_FREQ, RSFREQ


def preprocessing_cnt(file_path, l_freq=0.1, h_freq=75.0, sfreq=200):
    raw = mne.io.read_raw_cnt(file_path, preload=False)
    raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])

    raw.load_data()
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    raw = raw.resample(sfreq, n_jobs=5)
    eeg_data = raw.get_data(units='uV')

    return eeg_data, raw.ch_names


def process(cnt_file, out_dir, dataset_name):
    print(f'processing {cnt_file.name}')
    eeg_data, ch_order = preprocessing_cnt(cnt_file, L_FREQ, H_FREQ, RSFREQ)
    time_length = 4 * RSFREQ  # 4-second windows
    name = cnt_file.name.split('.')[0]
    for i in range(eeg_data.shape[1] // time_length):
        dump_path = os.path.join(out_dir, dataset_name, f'{dataset_name}_{name}_{i}.pkl')
        pickle.dump(
            {"X": eeg_data[:, i * time_length: (i + 1) * time_length],
             "ch_names": ch_order},
            open(dump_path, "wb"),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['SEED-FRA', 'SEED-GER'])
    parser.add_argument('--raw_dir', type=str, required=True, help='<French|German>/01-EEG-raw folder (*.cnt)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, args.dataset), exist_ok=True)
    group = list(Path(args.raw_dir).rglob('*.cnt'))
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(process, out_dir=args.out_dir, dataset_name=args.dataset), group)
