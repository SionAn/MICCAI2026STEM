"""Sort channels by the MNE standard montages and attach 3D electrode coordinates.

Reads <data_dir>/<name>/sub/... and writes <data_dir>/<name>/sorted/... with an extra 'ch_locations' field.
"""
import argparse
import os
import pickle

import mne
import numpy as np

from dataset_index import load_index

# Datasets whose pickles do not store (usable) channel names
FIXED_CHANNELS = {
    'BNCI2014_002': ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz',
                     'POz'],
    'BNCI2015_001': ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'],
    'BNCI2015_004': ['AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T3', 'C3', 'Cz', 'C4', 'T4',
                     'CP3', 'CPz', 'CP4', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO4',
                     'O1', 'O2'],
    'Liu2024': ['FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4', 'FT7', 'FT8', 'Cz', 'C3',
                'C4', 'T3', 'T4', 'CP3', 'CP4', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2'],
    'Schirrmeister2017': ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7',
                          'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
                          'POz', 'O1', 'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz',
                          'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3',
                          'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h',
                          'PO9', 'PO10', 'P9', 'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                          'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h',
                          'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h',
                          'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h',
                          'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h',
                          'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h'],
    'Weibo2014': ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                  'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz',
                  'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
                  'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6',
                  'PO8', 'O1', 'Oz', 'O2'],
    'Zhou2016': ['Fp1', 'Fp2', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'O1', 'Oz', 'O2'],
    'PhysioNet': ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2',
                  'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2',
                  'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
                  'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5',
                  'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
                  'O1', 'Oz', 'O2', 'Iz'],
    'ISRUC': ['F3', 'C3', 'O1', 'F4', 'C4', 'O2'],
}


def channel_sort(ch1, ch2, data):
    """Reorder channels ch2 (and rows of data) following the reference order ch1."""
    order_map = {name: i for i, name in enumerate(ch1)}
    sorted_chs = sorted(ch2, key=lambda ch: order_map.get(ch, len(order_map)))

    original_indices_map = {ch_name: i for i, ch_name in enumerate(ch2)}
    reorder_indices = [original_indices_map[ch_name] for ch_name in sorted_chs]
    sorted_data = data[reorder_indices, :]

    return sorted_chs, sorted_data


def get_channel_positions():
    channels = {}
    for montage in mne.channels.get_builtin_montages():
        channel = mne.channels.make_standard_montage(montage).get_positions()['ch_pos']
        for k in list(channel.keys()):
            v = channel.pop(k)
            channel[k.replace("'", "").upper()] = v
        channels = {**channels, **channel}
    return channels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    args = parser.parse_args()
    path = args.data_dir

    channels = get_channel_positions()
    ch_names = list(channels.keys())

    for d in args.datasets:
        dataset = load_index(path, d)
        for s_idx, s in enumerate(list(dataset.keys())):
            for c in list(dataset[s].keys()):
                for t_idx, t in enumerate(dataset[s][c]):
                    with open(os.path.join(path, d, 'sub', s, c, t), 'rb') as f:
                        data = pickle.load(f)
                    chs = FIXED_CHANNELS.get(d, data.get('ch_names'))
                    chs = [ch.upper().replace('CF', '').replace('CC', '').replace('EEG ', '').replace('SP', '')
                           for ch in chs]
                    sorted_chs, sorted_data = channel_sort(ch_names, chs, data['X'])
                    chs_location = np.stack([channels[ch] for ch in sorted_chs], axis=0)

                    os.makedirs(os.path.join(path, d, 'sorted', s, c), exist_ok=True)
                    out = {"X": sorted_data}
                    if 'Y' in data.keys():
                        out["Y"] = data['Y']
                    out["ch_names"] = sorted_chs
                    out["ch_locations"] = chs_location
                    pickle.dump(out, open(os.path.join(path, d, 'sorted', s, c, t), "wb"))
                    print(f'{d} - {s} ({s_idx + 1}/{len(dataset)}) - {c} - trial ({t_idx + 1}/{len(dataset[s][c])})',
                          end='\r')
        print('')
