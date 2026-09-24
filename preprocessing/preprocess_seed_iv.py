"""SEED-IV emotion dataset (pretraining).

Writes flat files to <out_dir>/SEED-IV/*.pkl; run regroup_subjects.py afterwards.
"""
import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mne
import scipy.io

from common import L_FREQ, H_FREQ, RSFREQ

DATASET_NAME = 'SEED-IV'

# Emotion label of the 24 trials in each session
SESSION_LABEL = {'1': [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                 '2': [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                 '3': [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]}

CH_NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
            'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
            'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
            'O2', 'CB2']


def preprocessing(file_path, l_freq=0.1, h_freq=75.0, sfreq=200):
    mat_file = scipy.io.loadmat(file_path)
    keys = [k for k in mat_file.keys() if 'eeg' in k]
    session = file_path.parent.name
    eeg_data = {}
    sr = 200
    for i in range(len(keys)):
        info = mne.create_info(ch_names=CH_NAMES, sfreq=sr, ch_types=['eeg'] * len(CH_NAMES))
        raw = mne.io.RawArray(mat_file[keys[i]], info)

        raw.load_data()
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
        raw = raw.notch_filter(50.0, n_jobs=5)
        raw = raw.resample(sfreq, n_jobs=5)
        eeg_data[i] = [raw.get_data(units='uV')[:, sr * 5:-45 * sr], SESSION_LABEL[session][i]]

    return eeg_data, raw.ch_names


def process(mat_file, out_dir):
    print(f'processing {mat_file.name}')
    eeg_data, ch_order = preprocessing(mat_file, L_FREQ, H_FREQ, RSFREQ)
    time_length = 4 * RSFREQ  # 4-second windows
    name = mat_file.name.split('.')[0]
    for s in range(24):
        for i in range(eeg_data[s][0].shape[1] // time_length):
            dump_path = os.path.join(out_dir, DATASET_NAME, f'{DATASET_NAME}_{name}_{s}_{i}.pkl')
            pickle.dump(
                {"X": eeg_data[s][0][:, i * time_length: (i + 1) * time_length],
                 "Y": eeg_data[s][1],
                 "ch_names": ch_order},
                open(dump_path, "wb"),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='SEED_IV/eeg_raw_data folder (contains session folders 1/, 2/, 3/)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, DATASET_NAME), exist_ok=True)
    group = list(Path(args.raw_dir).rglob('*.mat'))
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(process, out_dir=args.out_dir), group)
