"""ISRUC-Sleep subgroup 1 (downstream, 5-class sleep staging).

Uses a patched copy of MNE's EDF reader (edf_.py) to read the *.rec files.
Writes <out_dir>/ISRUC/sub/<subject id>/<label>/*.pkl directly.
"""
import argparse
import os
import pickle

import numpy as np

from edf_ import read_raw_edf

LABEL2ID = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}  # W, N1, N2, N3, REM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='ISRUC subgroup 1 folder (contains 1/1.rec, 1/1_1.txt, ..., 100/)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    args = parser.parse_args()

    for sub_id in range(1, 101):
        sub = str(sub_id)
        psg_f_name = os.path.join(args.raw_dir, sub, f'{sub}.rec')
        label_f_name = os.path.join(args.raw_dir, sub, f'{sub}_1.txt')

        raw = read_raw_edf(psg_f_name, preload=True)
        raw.filter(l_freq=0.1, h_freq=75)
        raw.notch_filter(50)
        raw.resample(sfreq=200)

        psg_array = raw.to_data_frame().values
        i = psg_array.shape[0] % (30 * 200)
        if i > 0:
            psg_array = psg_array[:-i, :]
        # same column selection as the CBraMod ISRUC preprocessing
        psg_array = psg_array[:, 2:8].reshape(-1, 30 * 200, 6)

        # keep a multiple of 20 epochs
        a = psg_array.shape[0] % 20
        if a > 0:
            psg_array = psg_array[:-a, :, :]

        labels_list = []
        for line in open(label_f_name).readlines():
            line_str = line.strip()
            if line_str != '':
                labels_list.append(LABEL2ID[line_str])
        label = np.array(labels_list)
        if a > 0:
            label = label[:-a]
        label = label.reshape(-1, 20).reshape(-1)
        data = psg_array.transpose(0, 2, 1)  # N, 6, 6000

        chs = raw.ch_names
        for count in range(data.shape[0]):
            data_dict = {
                'X': data[count],
                'Y': label[count],
                'ch_names': chs
            }
            dump_path = os.path.join(args.out_dir, 'ISRUC', 'sub', sub, str(label[count]))
            os.makedirs(dump_path, exist_ok=True)
            pickle.dump(data_dict, open(os.path.join(dump_path, f'ISRUC_{sub}_{count}.pkl'), 'wb'))
            print(f'{psg_f_name}: {count + 1}/{data.shape[0]}', end='\r')
