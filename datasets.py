import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

DATA_VERSION = 'sorted_scaled'  # output folder of preprocessing/scale.py


def build_index(data_dir, name):
    """Returns {subject: {class: [trial, ...]}} for <data_dir>/<name>/sorted_scaled (cached in <data_dir>/<name>.pkl)."""
    cache = os.path.join(data_dir, name + '.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)[name]

    root = os.path.join(data_dir, name, DATA_VERSION)
    index = {}
    for s in sorted(os.listdir(root)):
        index[s] = {c: sorted(os.listdir(os.path.join(root, s, c))) for c in sorted(os.listdir(os.path.join(root, s)))}
    with open(cache, 'wb') as f:
        pickle.dump({name: index}, f)
    return index


def seed_worker(worker_id):
    # numpy is not re-seeded per DataLoader worker by default
    np.random.seed(torch.initial_seed() % 2 ** 32)


class PretrainingDataset(Dataset):
    """Samples meta-learning episodes from the pre-training datasets.

    Each item contains, for a randomly chosen dataset and two subjects s1 != s2:
        x1: K+1 trials of s1 with class c1
        x2: K+1 trials of s1 with class c2 (c2 != c1)
        x3: one trial of s2 (random class)
        x4: one trial of s2 with class c1 (if available)
        y : 1 if c1 / c2 are valid task labels, 0 for unlabeled data (class 'Other')
    """

    def __init__(self, data_dir, datasets, n_sup=4, samples_per_dataset=10000):
        super().__init__()
        self.path = data_dir
        self.n_sup = n_sup
        self.samples_per_dataset = samples_per_dataset
        self.dataset_subject = {d: build_index(data_dir, d) for d in datasets}
        self.dataset = list(self.dataset_subject.keys())

    def __len__(self):
        return len(self.dataset) * self.samples_per_dataset

    def _load(self, dataset, sub, cls, trial):
        with open(os.path.join(self.path, dataset, DATA_VERSION, sub, cls, trial), 'rb') as f:
            file = pickle.load(f)  # X: (C, T)
        data = (file['X'] - file['X'].mean()) / file['X'].std()
        return data, file['ch_locations']

    def __getitem__(self, _):
        dataset = np.random.choice(self.dataset, 1, replace=False)[0]
        subjects = self.dataset_subject[dataset]
        sub = np.random.choice(list(subjects.keys()), 2, replace=False)
        same_subject_cls_list = list(subjects[sub[0]].keys())
        if len(same_subject_cls_list) == 1:
            cls1 = cls2 = 'Other'
            y = 0
        else:
            if 'Other' in same_subject_cls_list:
                same_subject_cls_list.remove('Other')
            cls1, cls2 = np.random.choice(same_subject_cls_list, 2, replace=False)
            y = 1

        same_subject_trial1 = np.random.choice(subjects[sub[0]][cls1], self.n_sup, replace=True)
        same_subject_trial2 = np.random.choice(subjects[sub[0]][cls2], self.n_sup, replace=True)

        diff_subject_cls_list = list(subjects[sub[1]].keys())
        diff_subject_cls = np.random.choice(diff_subject_cls_list, 1, replace=False)[0]
        diff_subject_trial = np.random.choice(subjects[sub[1]][diff_subject_cls], 1, replace=False)[0]

        if cls1 in diff_subject_cls_list:
            cls1_ = cls1
        else:
            cls1_ = diff_subject_cls
            y = 0
        diff_subject_same_cls = np.random.choice(subjects[sub[1]][cls1_], 1, replace=False)[0]

        x1, ch1 = zip(*[self._load(dataset, sub[0], cls1, t) for t in same_subject_trial1])
        x2, ch2 = zip(*[self._load(dataset, sub[0], cls2, t) for t in same_subject_trial2])
        x3, ch3 = self._load(dataset, sub[1], diff_subject_cls, diff_subject_trial)
        x4, ch4 = self._load(dataset, sub[1], cls1_, diff_subject_same_cls)

        return np.stack(x1, axis=-1), np.stack(x2, axis=-1), x3, x4, y, \
            np.stack(ch1, axis=-1), np.stack(ch2, axis=-1), ch3, ch4

    @staticmethod
    def collate(batch):
        """Pads samples with different numbers of channels and returns channel masks (1 = valid)."""
        x1, x2, x3, x4, y, ch1, ch2, ch3, ch4 = zip(*batch)
        length = torch.LongTensor([[a.shape[0], b.shape[0], c.shape[0], d.shape[0]]
                                   for a, b, c, d in zip(x1, x2, x3, x4)])

        def pad(items):
            return pad_sequence([torch.from_numpy(i) for i in items], batch_first=True)

        x1, x2, x3, x4 = pad(x1), pad(x2), pad(x3), pad(x4)  # (B, C, T, S) / (B, C, T)
        ch1, ch2, ch3, ch4 = pad(ch1), pad(ch2), pad(ch3), pad(ch4)

        largest_c = max(x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1])
        pad4 = lambda t: torch.nn.functional.pad(t, [0, 0, 0, 0, 0, largest_c - t.shape[1]])
        pad3 = lambda t: torch.nn.functional.pad(t, [0, 0, 0, largest_c - t.shape[1]])
        x1, x2, x3, x4 = pad4(x1), pad4(x2), pad3(x3), pad3(x4)
        ch1, ch2, ch3, ch4 = pad4(ch1), pad4(ch2), pad3(ch3), pad3(ch4)

        indices = torch.arange(largest_c)
        masks = [(indices < length[:, i].unsqueeze(1)).long() for i in range(4)]

        return x1.float(), x2.float(), x3.float(), x4.float(), torch.LongTensor(y).unsqueeze(1), *masks, \
            ch1.float() * 10, ch2.float() * 10, ch3.float() * 10, ch4.float() * 10


def subject_split(name, subject):
    """Cross-subject train / val / test split of the downstream datasets."""
    if name == 'PhysioNet':  # e.g. S001
        idx = int(subject.split('S')[1])
        bounds = (70, 89)
    elif name == 'SHU':  # e.g. sub-001_ses-01_task_motorimagery_eeg.mat
        idx = int(subject.split('_')[0].split('-')[1])
        bounds = (16, 21)
    elif name == 'ISRUC':  # e.g. 1
        idx = int(subject)
        bounds = (81, 91)
    else:
        raise ValueError(f'Unknown downstream dataset: {name}')
    if idx < bounds[0]:
        return 'train'
    return 'val' if idx < bounds[1] else 'test'


class DownstreamDataset(Dataset):
    """Downstream dataset for fine-tuning STEM.

    Validation / test items: (x, y, ch).
    Training items additionally contain the meta-learning samples:
        z  : one trial of the same subject for each other class (task negatives / subject positives)
        p  : one trial of a different subject with the same class (task positive / subject negative)
    """

    def __init__(self, data_dir, name, split, seq_len, patch_size, num_cls, meta=True):
        super().__init__()
        self.path = data_dir
        self.name = name
        self.split = split
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_cls = num_cls
        self.meta = meta and split == 'train'

        index = build_index(data_dir, name)
        self.subjects = {s: v for s, v in index.items() if subject_split(name, s) == split}
        self.samples = [(s, c, t) for s, classes in self.subjects.items() for c, trials in classes.items()
                        for t in trials]
        print(f'{name} {split}: {len(self.subjects)} subjects, {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def load(self, sub, cls, trial):
        with open(os.path.join(self.path, self.name, DATA_VERSION, sub, cls, trial), 'rb') as f:
            file = pickle.load(f)
        data = file['X'] - file['X'].mean(axis=1, keepdims=True)
        data = data.reshape(-1, self.seq_len, self.patch_size).astype(np.float32)  # (C, P, L)
        ch = (file['ch_locations'] * 10).astype(np.float32)  # (C, 3)
        return data, ch

    def __getitem__(self, idx):
        sub, cls, trial = self.samples[idx]
        data, ch = self.load(sub, cls, trial)
        y = int(cls)
        if not self.meta:
            return data, y, ch

        # different subject, same class
        other_subjects = [s for s in self.subjects if s != sub]
        while True:
            diff_sub = np.random.choice(other_subjects)
            if cls in self.subjects[diff_sub]:
                break
        data_p, ch_p = self.load(diff_sub, cls, np.random.choice(self.subjects[diff_sub][cls]))

        # same subject, other classes (repeated if the subject does not have all classes)
        z_cls_list = [c for c in self.subjects[sub] if c != cls]
        z_cls_list = [z_cls_list[i % len(z_cls_list)] for i in range(self.num_cls - 1)]
        data_z, ch_z = zip(*[self.load(sub, c, np.random.choice(self.subjects[sub][c])) for c in z_cls_list])

        return data, y, ch, np.stack(data_z, axis=0), np.stack(ch_z, axis=0), data_p, ch_p


class FewShotDataset(DownstreamDataset):
    """Returns each query together with K' support trials per class from the same subject.

    Items: (x, y, ch, z, ch_z) with z: (num_cls, K', C, P, L). The query itself is never used as support.
    If a subject has no trial of some class, the support set of a random available class is reused.
    """

    def __init__(self, data_dir, name, split, seq_len, patch_size, num_cls, few):
        super().__init__(data_dir, name, split, seq_len, patch_size, num_cls, meta=False)
        self.few = few

    def __getitem__(self, idx):
        sub, cls, trial = self.samples[idx]
        data, ch = self.load(sub, cls, trial)
        y = int(cls)

        support = {}
        for c, trials in self.subjects[sub].items():
            candidates = [t for t in trials if t != trial] or trials
            chosen = np.random.choice(candidates, size=self.few, replace=True)
            x_s, ch_s = zip(*[self.load(sub, c, t) for t in chosen])
            support[int(c)] = (np.stack(x_s, axis=0), np.stack(ch_s, axis=0))
        available = list(support.keys())
        for c in range(self.num_cls):
            if c not in support:
                support[c] = support[available[np.random.randint(len(available))]]

        data_z = np.stack([support[c][0] for c in range(self.num_cls)], axis=0)
        ch_z = np.stack([support[c][1] for c in range(self.num_cls)], axis=0)
        return data, y, ch, data_z, ch_z
