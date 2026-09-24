"""Reorganize flat outputs (<data_dir>/<name>/*.pkl) into <data_dir>/<name>/sub/<subject>/<class>/*.pkl.

Needed for: BCI-IV-1, EmoBrain, GAL, Siena, TUEG, SPIS, Raw, Rest, SEED-IV, SEED-FRA, SEED-GER.
Unlabeled samples are put in the class folder 'Other'.
"""
import argparse
import glob
import os
import pickle
import shutil

from tqdm import tqdm

FLAT_DATASETS = ['BCI-IV-1', 'EmoBrain', 'GAL', 'Siena', 'TUEG', 'SPIS', 'Raw', 'Rest',
                 'SEED-IV', 'SEED-FRA', 'SEED-GER']
# Datasets whose labeled files carry an extra trial index (<name>_<recording>_<trial>_<window>.pkl)
TRIAL_INDEXED = ['BCI-IV-1', 'EmoBrain', 'SEED-IV']


def strip_last(name):
    # 'A_B_C' -> 'A_B'
    return name[:-1 - len(name.split('_')[-1])]


def split_by_class(path, d):
    files = glob.glob(os.path.join(path, d, '*.pkl'))
    for file in tqdm(files, desc=d):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        cls = str(data['Y']) if 'Y' in data.keys() else 'Other'
        os.makedirs(os.path.join(path, d, cls), exist_ok=True)
        shutil.move(file, os.path.join(path, d, cls, os.path.basename(file)))


def split_by_subject(path, d):
    classes = [c for c in os.listdir(os.path.join(path, d)) if c != 'sub']
    for c in classes:
        files = glob.glob(os.path.join(path, d, c, '*.pkl'))
        for count, f in enumerate(files):
            name = os.path.basename(f)
            sub = strip_last(name)
            os.makedirs(os.path.join(path, d, 'sub', sub, c), exist_ok=True)
            shutil.move(f, os.path.join(path, d, 'sub', sub, c, name))
            print(f'Dataset: {d} - Class: {c} - Length: {count + 1}/{len(files)}', end='\r')
        os.rmdir(os.path.join(path, d, c))


def merge_trials(path, d):
    folders = os.listdir(os.path.join(path, d, 'sub'))
    if d == 'BCI-IV-1':
        folders = [j for j in folders if 'calib' in j]
    if d == 'EmoBrain':
        folders = [j for j in folders if len(j.split('_')) == 8]
    for count, folder in enumerate(folders):
        fold = strip_last(folder)
        for f in glob.glob(os.path.join(path, d, 'sub', folder, '*', '*.pkl')):
            cls = f.split('/')[-2]
            os.makedirs(os.path.join(path, d, 'sub', fold, cls), exist_ok=True)
            shutil.move(f, os.path.join(path, d, 'sub', fold, cls, os.path.basename(f)))
        # remove the now-empty per-trial folder so it is not indexed as a subject
        for cls in os.listdir(os.path.join(path, d, 'sub', folder)):
            os.rmdir(os.path.join(path, d, 'sub', folder, cls))
        os.rmdir(os.path.join(path, d, 'sub', folder))
        print(f'Dataset: {d} - Length: {count + 1}/{len(folders)}', end='\r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='root of the processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=FLAT_DATASETS, choices=FLAT_DATASETS)
    args = parser.parse_args()

    for d in args.datasets:
        split_by_class(args.data_dir, d)
        split_by_subject(args.data_dir, d)
        if d in TRIAL_INDEXED:
            merge_trials(args.data_dir, d)
        print('')
