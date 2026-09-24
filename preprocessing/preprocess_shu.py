"""SHU-MI dataset (downstream, 2-class motor imagery).

Writes <out_dir>/SHU/sub/<session file>/<label>/*.pkl directly.
"""
import argparse
import os
import pickle

import scipy.io
from scipy import signal

CH_NAMES = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz',
            'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4',
            'T5', 'T6', 'PO3', 'PO4', 'Oz', 'O1', 'O2']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help='SHU-MI mat folder (sub-XXX_ses-XX_task_motorimagery_eeg.mat)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    args = parser.parse_args()

    for file in sorted(os.listdir(args.raw_dir)):
        data = scipy.io.loadmat(os.path.join(args.raw_dir, file))
        eeg = data['data']
        labels = data['labels'][0]
        eeg_resample = signal.resample(eeg, 800, axis=2)  # 4 s at 200 Hz
        for i, (sample, label) in enumerate(zip(eeg_resample, labels)):
            data_dict = {
                'X': sample,
                'Y': label - 1,
                'ch_names': CH_NAMES
            }
            dump_path = os.path.join(args.out_dir, 'SHU', 'sub', file, str(data_dict['Y']))
            os.makedirs(dump_path, exist_ok=True)
            pickle.dump(data_dict, open(os.path.join(dump_path, f'SHU_{file[:-4]}-{i}.pkl'), 'wb'))
            print(f'{file}: {i + 1}/{eeg_resample.shape[0]}', end='\r')
