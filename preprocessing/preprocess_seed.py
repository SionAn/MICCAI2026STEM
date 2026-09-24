"""SEED emotion dataset (pretraining).

Writes <out_dir>/SEED/sub/<recording>/<label>/*.pkl directly (no regrouping needed).
"""
import argparse
import os
import pickle

import mne

USELESS_CH = ['M1', 'M2', 'VEO', 'HEO']
# Start / end (in seconds) of the 15 film clips in each recording
TRIALS = {
    'start': [27, 290, 551, 784, 1050, 1262, 1484, 1748, 1993, 2287, 2551, 2812, 3072, 3335, 3599],
    'end': [262, 523, 757, 1022, 1235, 1457, 1721, 1964, 2258, 2524, 2786, 3045, 3307, 3573, 3805]
}
LABELS = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]  # 0: sad, 1: neutral, 2: happy

DATASET_NAME = 'SEED'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True, help='SEED_RAW_EEG folder (*.cnt)')
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    args = parser.parse_args()

    files = sorted(os.listdir(args.raw_dir))
    for file in files:
        raw = mne.io.read_raw_cnt(os.path.join(args.raw_dir, file), preload=True)
        raw.drop_channels(USELESS_CH)
        raw.resample(200)
        raw.filter(l_freq=0.1, h_freq=75, n_jobs=5)
        raw = raw.notch_filter(50.0, n_jobs=5)
        data_matrix = raw.get_data(units='uV')
        data_trials = [data_matrix[:, TRIALS['start'][j] * 200:TRIALS['end'][j] * 200] for j in range(15)]
        chs = raw.ch_names
        for trial in range(len(data_trials)):
            data = data_trials[trial].reshape(62, -1, 200)
            dump_path = os.path.join(args.out_dir, DATASET_NAME, 'sub', file[:-4], str(LABELS[trial]))
            for count in range(int(data.shape[1] // 4)):
                os.makedirs(dump_path, exist_ok=True)
                data_dict = {
                    'X': data[:, count * 4: (count + 1) * 4].reshape(62, 800),
                    'Y': LABELS[trial],
                    'ch_names': chs
                }
                pickle.dump(data_dict, open(os.path.join(dump_path, f'SEED_{file[:-4]}_{trial}_{count}.pkl'), 'wb'))
            print(f'{file}: trial {trial + 1}/{len(data_trials)}', end='\r')
