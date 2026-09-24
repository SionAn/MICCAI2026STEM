import os
import pickle


def load_index(path, d):
    """Return {subject: {class: [trial files]}} for <path>/<d>/sub, cached in <path>/<d>.pkl."""
    index_path = os.path.join(path, d + '.pkl')
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            return pickle.load(f)[d]
    dataset_subject_class = {}
    for s in os.listdir(os.path.join(path, d, 'sub')):
        dataset_subject_class_trial = {}
        for c in os.listdir(os.path.join(path, d, 'sub', s)):
            dataset_subject_class_trial[c] = os.listdir(os.path.join(path, d, 'sub', s, c))
        dataset_subject_class[s] = dataset_subject_class_trial
    pickle.dump({d: dataset_subject_class}, open(index_path, "wb"))
    return dataset_subject_class
