"""Siena Scalp EEG Database (pretraining). Label 1: seizure, 0: non-seizure.

Writes flat files to <out_dir>/Siena/*.pkl; run regroup_subjects.py afterwards.
"""
import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mne

from common import STANDARD_1020, L_FREQ, H_FREQ, RSFREQ

DATASET_NAME = 'Siena'

# Seizure intervals (start, end) in seconds for each recording
SESSION_LABEL = {
    'PN00-1.edf': [[1143, 1213]],
    'PN00-2.edf': [[1220, 1274]],
    'PN00-3.edf': [[765, 44525]],
    'PN00-4.edf': [[1006, 1080]],
    'PN00-5.edf': [[904, 971]],
    'PN01.edf': [[10218, 10272]],
    'PN03-1.edf': [[38673, 38784]],
    'PN03-2.edf': [[34921, 35054]],
    'PN05-2.edf': [[7163, 7198]],
    'PN05-3.edf': [[6836, 6866]],
    'PN05-4.edf': [[3608, 3647]],
    'PN06-1.edf': [[5583, 5647]],
    'PN06-2.edf': [[8860, 8929]],
    'PN06-3.edf': [[6275, 6317]],
    'PN06-4.edf': [[5939, 6002]],
    'PN06-5.edf': [[4783, 4827]],
    'PN07 crisi 1.edf': [[22059, 22121]],
    'PN09-1.edf': [[7249, 7329]],
    'PN09-2.edf': [[7127, 7186]],
    'PN09-3.edf': [[7221, 7285]],
    'PN10-1.edf': [[7545, 7614]],
    'PN10-2.edf': [[7798, 7849]],
    'PN10-3.edf': [[7835, 7904]],
    'PN10-4.5.6.edf': [[2309, 2314], [6544, 6563], [11225, 11282]],
    'PN10-7.8.9.edf': [[38748, 38796], [5459, 5477], [12923, 12938]],
    'PN10-10.edf': [[7977, 7991]],
    'PN11-.edf': [[7554, 7609]],
    'PN12-1.2.edf': [[1312, 1375], [9570, 9638]],
    'PN12-3.edf': [[772, 868]],
    'PN12-4.edf': [[9812, 9875]],
    'PN13-1.edf': [[7062, 7110]],
    'PN13-2.edf': [[7249, 7314]],
    'PN13-3.edf': [[7553, 7704]],
    'PN14-1.edf': [[7262, 7289]],
    'PN14-2.edf': [[7479, 7491]],
    'PN14-3.edf': [[17540, 17581]],
    'PN14-4.edf': [[5463, 5546]],
    'PN16-1.edf': [[7184, 7307]],
    'PN16-2.edf': [[8574, 8681]],
    'PN17-1.edf': [[8420, 8490]],
    'PN17-2.edf': [[7731, 7814]]
}


def preprocessing_edf(file_path, l_freq=0.1, h_freq=75.0, sfreq=200):
    raw = mne.io.read_raw_edf(file_path, preload=False)
    drop_channels = []
    for ch in raw.ch_names:
        if ch == 'SPO2':
            ch = ch.replace('SP', 'EEG ')
        if ch not in STANDARD_1020 and 'EEG' not in ch:
            drop_channels.append(ch)
    raw.drop_channels(drop_channels)

    raw.load_data()
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(50.0, n_jobs=5)
    raw = raw.resample(sfreq, n_jobs=5)
    eeg_data = raw.get_data(units='uV')

    return eeg_data, raw.ch_names


def dump_windows(signal, label, ch_order, out_dir, name, num):
    time_length = 4 * RSFREQ  # 4-second windows
    for i in range(signal.shape[1] // time_length):
        dump_path = os.path.join(out_dir, DATASET_NAME, f'{DATASET_NAME}_{name}_{num}.pkl')
        pickle.dump(
            {"X": signal[:, i * time_length: (i + 1) * time_length],
             "Y": label,
             "ch_names": ch_order},
            open(dump_path, "wb"),
        )
        num += 1
    return num


def process(edf_file, out_dir):
    print(f'processing {edf_file.name}')
    eeg_data, ch_order = preprocessing_edf(edf_file, L_FREQ, H_FREQ, RSFREQ)
    eeg_data = eeg_data[:, :-10 * RSFREQ]
    name = edf_file.name.split('.')[0]

    label = SESSION_LABEL[edf_file.name]
    num = 0
    for t in range(len(label)):
        # non-seizure segment before the first seizure
        if t == 0:
            signal = eeg_data[:, :label[t][0] * RSFREQ]
        else:
            signal = eeg_data[:, label[t - 1][1] * RSFREQ:label[t][0] * RSFREQ]
        num = dump_windows(signal, 0, ch_order, out_dir, name, num)

        # seizure segment
        signal = eeg_data[:, label[t][0] * RSFREQ:label[t][1] * RSFREQ]
        num = dump_windows(signal, 1, ch_order, out_dir, name, num)

        # non-seizure segment after the seizure
        if len(label) == 1 or (t + 1) == len(label):
            signal = eeg_data[:, label[t][1] * RSFREQ:]
        else:
            signal = eeg_data[:, label[t][1] * RSFREQ:label[t + 1][0] * RSFREQ]
        num = dump_windows(signal, 0, ch_order, out_dir, name, num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help='Siena Scalp EEG Database folder (*.edf)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, DATASET_NAME), exist_ok=True)
    group = list(Path(args.raw_dir).rglob('*.edf'))
    with Pool(processes=args.num_workers) as pool:
        pool.map(partial(process, out_dir=args.out_dir), group)
