"""MOABB motor imagery datasets (pretraining), loaded through braindecode.

Raw data are downloaded automatically by MOABB (to ~/mne_data by default).
Writes <out_dir>/<dataset>/sub/<subject>_<session>/<label>/*.pkl directly (no regrouping needed).
Channel names are not stored here; assign_channel_locations.py uses a fixed list per dataset.
"""
import argparse
import os
import pickle

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from numpy import multiply

SUBJECT_LIST = {
    'BNCI2014_002': list(range(1, 15)),
    'BNCI2015_001': list(range(1, 13)),
    'BNCI2015_004': list(range(1, 10)),
    'Liu2024': list(range(1, 51)),
    'Schirrmeister2017': list(range(1, 15)),
    'Weibo2014': list(range(1, 11)),
    'Zhou2016': list(range(1, 5)),
}

# preprocessing parameters
L_FREQ = 0.1
H_FREQ = 75.0
RSFREQ = 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=list(SUBJECT_LIST.keys()))
    parser.add_argument('--out_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--n_jobs', type=int, default=8)
    args = parser.parse_args()

    dataset_name = args.dataset
    print('Processing: ', dataset_name)
    os.makedirs(os.path.join(args.out_dir, dataset_name), exist_ok=True)

    for sub in SUBJECT_LIST[dataset_name]:
        dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=sub)
        try:
            preprocessing = [
                Preprocessor('pick_types', eeg=True, meg=False, stim=False),
                Preprocessor('filter', l_freq=L_FREQ, h_freq=H_FREQ),
                Preprocessor('resample', sfreq=RSFREQ),
                Preprocessor(lambda data: multiply(data, 1e6))  # V -> uV
            ]
            dataset_session = dataset.split('session')
            for sess, split in dataset_session.items():
                preprocess(split, preprocessing, n_jobs=args.n_jobs)
                window = create_windows_from_events(split)
                time_length = 4 * RSFREQ
                num = 0
                for x, y, _ in window:
                    dump_dir = os.path.join(args.out_dir, dataset_name, 'sub', f'{sub}_{sess}', str(y))
                    os.makedirs(dump_dir, exist_ok=True)
                    pickle.dump(
                        {"X": x[:, -time_length:],
                         "Y": y},
                        open(os.path.join(dump_dir, f'{dataset_name}_{sub}_{sess}_{num}.pkl'), "wb"),
                    )
                    num += 1
        except Exception as e:
            print('Error in Subject', sub, 'on', dataset_name, ':', e)
