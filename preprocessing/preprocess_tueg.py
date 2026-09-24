"""TUH EEG Corpus (TUEG v2.0.1, pretraining, unlabeled).

Adapted from NeuroLM (https://github.com/935963004/NeuroLM).
Writes flat files to <out_dir>/TUEG/*.pkl; run regroup_subjects.py afterwards.
"""
import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mne

from common import STANDARD_1020, L_FREQ, H_FREQ, RSFREQ

DATASET_NAME = 'TUEG'


def preprocessing_edf(file_path, l_freq=0.1, h_freq=75.0, sfreq=200):
    raw = mne.io.read_raw_edf(file_path, preload=False)
    # skip linked-ear referenced recordings
    if raw.ch_names[0].split('-')[-1] == 'LE':
        return None, raw.ch_names
    raw.drop_channels([ch for ch in raw.ch_names if ch.split(' ')[-1].split('-')[0] not in STANDARD_1020])

    raw.load_data()
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(60.0, n_jobs=5)
    raw = raw.resample(sfreq, n_jobs=5)
    eeg_data = raw.get_data(units='uV')

    return eeg_data, raw.ch_names


def process(edf_file, out_dir):
    print(f'processing {edf_file.name}')
    eeg_data, ch_order = preprocessing_edf(edf_file, L_FREQ, H_FREQ, RSFREQ)
    if eeg_data is None:
        return

    ch_order = [s.split(' ')[-1].split('-')[0] for s in ch_order]
    eeg_data = eeg_data[:, :-10 * RSFREQ]
    time_length = 4 * RSFREQ  # 4-second windows
    name = edf_file.name.split('.')[0]
    for i in range(eeg_data.shape[1] // time_length):
        dump_path = os.path.join(out_dir, DATASET_NAME, f'{DATASET_NAME}_{name}_{i}.pkl')
        pickle.dump(
            {"X": eeg_data[:, i * time_length: (i + 1) * time_length], "ch_names": ch_order},
            open(dump_path, "wb"),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help='tuh_eeg/v2.0.1/edf folder')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--num_workers', type=int, default=24)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, DATASET_NAME), exist_ok=True)
    group = list(Path(args.raw_dir).rglob('*.edf'))
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(process, out_dir=args.out_dir), group)
