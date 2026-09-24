"""SPIS resting-state dataset (pretraining, unlabeled).

Writes flat files to <out_dir>/SPIS/*.pkl; run regroup_subjects.py afterwards.
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

DATASET_NAME = 'SPIS'

CH_NAMES = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
            'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
            'Fpz', 'Fp2', 'AF8', 'AF4', 'Afz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
            'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']


def preprocessing_mat(file_path, l_freq=0.1, h_freq=75.0, sfreq=200):
    mat_data = scipy.io.loadmat(file_path)
    eeg_data_uv = mat_data['dataRest'][:64] * 1e-5
    sr = 256
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sr, ch_types=['eeg'] * len(CH_NAMES))
    raw = mne.io.RawArray(eeg_data_uv, info)

    raw.load_data()
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    raw = raw.resample(sfreq, n_jobs=5)
    eeg_data = raw.get_data(units='uV')

    return eeg_data, raw.ch_names


def process(mat_file, out_dir):
    print(f'processing {mat_file.name}')
    eeg_data, ch_order = preprocessing_mat(mat_file, L_FREQ, H_FREQ, RSFREQ)
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
    parser.add_argument('--raw_dir', type=str, required=True, help='SPIS resting-state folder (*.mat)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, DATASET_NAME), exist_ok=True)
    group = list(Path(args.raw_dir).rglob('*.mat'))
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(process, out_dir=args.out_dir), group)
