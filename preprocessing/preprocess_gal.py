"""WAY-EEG-GAL grasp-and-lift dataset (pretraining).

Adapted from NeuroLM (https://github.com/935963004/NeuroLM).
Writes flat files to <out_dir>/GAL/*.pkl; run regroup_subjects.py afterwards.
"""
import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mne
import numpy as np
import scipy.io

from common import STANDARD_1020, L_FREQ, H_FREQ, RSFREQ

DATASET_NAME = 'GAL'


def preprocessing(file_path, l_freq=0.1, h_freq=75.0, sfreq=200):
    mat_data = scipy.io.loadmat(file_path)
    sr = 500
    signal = []
    for i in range(mat_data['ws'][0][0][4]['eeg'].shape[1]):
        seq = mat_data['ws'][0][0][4]['eeg'][0, i]  # T, 32
        signal.append(seq[:sr * (seq.shape[0] // sr)])
    signal = np.concatenate(signal, axis=0).T

    ch_names = []
    for i in range(mat_data['ws'][0][0][3]['eeg'][0][0][0].shape[0]):
        ch_names.append(mat_data['ws'][0][0][3]['eeg'][0][0][0][i][0])

    info = mne.create_info(ch_names=ch_names, sfreq=sr, ch_types=['eeg'] * len(ch_names))
    raw = mne.io.RawArray(signal, info)
    raw.drop_channels([ch for ch in raw.ch_names if ch not in STANDARD_1020])

    raw.load_data()
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    raw = raw.resample(sfreq, n_jobs=5)
    eeg_data = raw.get_data(units='uV')

    return eeg_data, raw.ch_names


def process(mat_file, out_dir):
    # only the WS_* (window series) files are used
    if 'WS' not in mat_file.name:
        return
    print(f'processing {mat_file.name}')
    eeg_data, ch_order = preprocessing(mat_file, L_FREQ, H_FREQ, RSFREQ)
    time_length = 4 * RSFREQ  # 4-second windows
    name = mat_file.name.split('.')[0]
    for i in range(eeg_data.shape[1] // time_length):
        dump_path = os.path.join(out_dir, DATASET_NAME, f'{DATASET_NAME}_{name}_{i}.pkl')
        pickle.dump(
            {"X": eeg_data[:, i * time_length: (i + 1) * time_length],
             "ch_names": ch_order},
            open(dump_path, "wb"),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help='folder containing the GAL *.mat files')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, DATASET_NAME), exist_ok=True)
    group = list(Path(args.raw_dir).rglob('*.mat'))
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(process, out_dir=args.out_dir), group)
