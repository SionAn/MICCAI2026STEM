import os
import random

import numpy as np
import torch
from sklearn.metrics import (auc, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

# Pre-training datasets cited in the paper (folder names under --data_dir).
PRETRAIN_DATASETS = [
    # motor imagery
    'BCI-IV-1', 'BNCI2014_002', 'BNCI2015_001', 'BNCI2015_004', 'Liu2024', 'Schirrmeister2017', 'Weibo2014',
    'Zhou2016',
    # emotion recognition
    'EmoBrain', 'SEED', 'SEED-IV',
    # seizure detection
    'Siena',
    # others
    'SEED-FRA', 'SEED-GER', 'GAL', 'TUEG', 'SPIS', 'Raw', 'Rest',
]

# Downstream datasets: number of classes, channels and sample length (at 200 Hz).
DOWNSTREAM_DATASETS = {
    'PhysioNet': {'num_cls': 4, 'num_ch': 64, 'length': 800},
    'SHU': {'num_cls': 2, 'num_ch': 32, 'length': 800},
    'ISRUC': {'num_cls': 5, 'num_ch': 6, 'length': 6000},
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_state_dict(path):
    """Loads a state dict saved either as a plain state dict or as a training checkpoint."""
    state_dict = torch.load(path, map_location='cpu')
    if 'model' in state_dict and isinstance(state_dict['model'], dict):
        state_dict = state_dict['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # unused parameters of the original implementation
    for k in ['start_embedding', 'end_embedding']:
        state_dict.pop(k, None)
    return state_dict


def load_pretrained(model, path):
    results = model.load_state_dict(load_state_dict(path), strict=False)
    print(f'Loaded pre-trained weights from {path}')
    print(f'  newly initialized: {sorted(results.missing_keys)}')
    if results.unexpected_keys:
        print(f'  unused: {sorted(results.unexpected_keys)}')
    return model


def compute_metrics(truths, scores, binary):
    """Balanced accuracy + (Cohen's kappa, weighted F1) or (AUROC, AUCPR) for binary tasks."""
    preds = scores.argmax(axis=1)
    metrics = {'acc': balanced_accuracy_score(truths, preds), 'cm': confusion_matrix(truths, preds)}
    if binary:
        metrics['auroc'] = roc_auc_score(truths, scores[:, 1])
        precision, recall, _ = precision_recall_curve(truths, scores[:, 1], pos_label=1)
        metrics['aucpr'] = auc(recall, precision)
    else:
        metrics['kappa'] = cohen_kappa_score(truths, preds)
        metrics['f1'] = f1_score(truths, preds, average='weighted')
    return metrics


def selection_score(metrics):
    """Model selection criterion: Kappa for multi-class tasks and AUCPR for binary tasks."""
    return metrics['aucpr'] if 'aucpr' in metrics else metrics['kappa']


def format_metrics(metrics):
    keys = ['acc', 'aucpr', 'auroc'] if 'aucpr' in metrics else ['acc', 'kappa', 'f1']
    return ', '.join(f'{k}: {metrics[k]:.5f}' for k in keys)


def makedirs(path):
    if path:
        os.makedirs(path, exist_ok=True)
