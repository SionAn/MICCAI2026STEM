"""BCI Competition IV dataset 1 (pretraining).

Adapted from NeuroLM (https://github.com/935963004/NeuroLM).
Writes flat files to <out_dir>/BCI-IV-1/*.pkl; run regroup_subjects.py afterwards.
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

DATASET_NAME = 'BCI-IV-1'


def preprocessing_mat(file_path, l_freq=0.1, h_freq=75.0, sfreq=200):
    mat_data = scipy.io.loadmat(file_path)
    cnt = mat_data['cnt']
    nfo = mat_data['nfo']
    if 'mrk' in mat_data.keys():
        mrk = mat_data['mrk']
        pos = mrk['pos'][0, 0][0]
        label = mrk['y'][0, 0][0]
    else:
        pos = []
        label = []

    sr = nfo['fs'][0, 0][0, 0]
    ch_names = [s[0] for s in nfo['clab'][0][0][0]]
    eeg_data_uv = (cnt.astype(float) * 0.1).T
    info = mne.create_info(ch_names=ch_names, sfreq=sr, ch_types=['eeg'] * len(ch_names))
    raw = mne.io.RawArray(eeg_data_uv, info)

    raw.load_data()
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    raw = raw.resample(sfreq, n_jobs=5)
    eeg_data = raw.get_data(units='uV')

    return eeg_data, raw.ch_names, pos, label


def process(cnt_file, out_dir):
    print(f'processing {cnt_file.name}')
    eeg_data, ch_order, pos, label = preprocessing_mat(cnt_file, L_FREQ, H_FREQ, RSFREQ)
    time_length = 4 * RSFREQ  # 4-second windows
    name = cnt_file.name.split('.')[0]
    if len(label) == 0:
        eeg_data = eeg_data[:, :-10 * RSFREQ]
        for i in range(eeg_data.shape[1] // time_length):
            dump_path = os.path.join(out_dir, DATASET_NAME, f'{DATASET_NAME}_{name}_{i}.pkl')
            pickle.dump(
                {"X": eeg_data[:, i * time_length: (i + 1) * time_length],
                 "ch_names": ch_order},
                open(dump_path, "wb"),
            )
    else:
        for s in range(len(label)):
            start = int(pos[s] / 5)
            end = start + 4 * RSFREQ  # 4-second motor imagery task
            for i in range((end - start) // time_length):
                dump_path = os.path.join(out_dir, DATASET_NAME, f'{DATASET_NAME}_{name}_{s}_{i}.pkl')
                pickle.dump(
                    {"X": eeg_data[:, start + i * time_length: start + (i + 1) * time_length],
                     "Y": label[s],
                     "ch_names": ch_order},
                    open(dump_path, "wb"),
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help='folder containing the BCI IV-1 *.mat files')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, DATASET_NAME), exist_ok=True)
    group = list(Path(args.raw_dir).rglob('*.mat'))
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(process, out_dir=args.out_dir), group)
