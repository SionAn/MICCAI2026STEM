"""Rescale every sample by powers of 10 so that max |X| lies in [0.3, 3].

Reads <data_dir>/<name>/sorted/... and writes <data_dir>/<name>/sorted_scaled/..., which is used for training.
"""
import argparse
import os
import pickle

from dataset_index import load_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    args = parser.parse_args()
    path = args.data_dir

    for d in args.datasets:
        dataset = load_index(path, d)
        for s_idx, s in enumerate(list(dataset.keys())):
            for c in list(dataset[s].keys()):
                for t_idx, t in enumerate(dataset[s][c]):
                    with open(os.path.join(path, d, 'sorted', s, c, t), 'rb') as f:
                        data = pickle.load(f)
                    while abs(data['X']).max() > 3:
                        data['X'] /= 10
                    while abs(data['X']).max() < 0.3:
                        data['X'] *= 10
                    os.makedirs(os.path.join(path, d, 'sorted_scaled', s, c), exist_ok=True)
                    out = {"X": data['X']}
                    if 'Y' in data.keys():
                        out["Y"] = data['Y']
                    out["ch_names"] = data['ch_names']
                    out["ch_locations"] = data['ch_locations']
                    pickle.dump(out, open(os.path.join(path, d, 'sorted_scaled', s, c, t), "wb"))
                    print(f'{d} - {s} ({s_idx + 1}/{len(dataset)}) - {c} - trial ({t_idx + 1}/{len(dataset[s][c])})',
                          end='\r')
        print('')
