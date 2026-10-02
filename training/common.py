"""
Shared definitions for the training/ scripts.

Everything here is used by more than one stage: the output layout, the CNN
backbones, the dataset pipeline, the experiment grid and the Excel writers.
Keeping one copy matters beyond tidiness -- every stage has to resolve a grid
position to the same train/val/test indices, and fold_splits() is the single
definition they all call. It honours the official train/test protocol of the
parent datasets: training and validation samples come from the official train
split, test samples from the official test split, never mixed.

Deliberately free of qiskit imports, so it can be imported from both
environment A (Qiskit 1.x) and environment B (Qiskit 2.x).
"""

import os

import numpy as np
import pandas as pd
import joblib
import openpyxl
from openpyxl import Workbook

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# Anchored to the repo root (the code/ directory holding training/), so every
# stage reads and writes the same tree no matter which working directory the
# script is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULT_ROOT = os.path.join(REPO_ROOT, "results")
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")

# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_ARCHS = ["lightweight", "complex"]

# Architecture configuration.
#   model_name : output directory prefix; the QSVM and QPU stages locate the
#                checkpoints through it, so it must not drift between stages.
#   dropout    : dropout rate of the MLP classification head.
ARCH_CONFIG = {
    "lightweight": dict(model_name="Lightweight_BatchNorm", dropout=0.1),
    "complex":     dict(model_name="complex_BatchNorm",     dropout=0.3),
}

# Dataset codes; the trailing digits are the two class indices of the parent
# dataset (see README).
DATASET_NAMES = ["Fashion_59", "Fashion_24", "CIFAR_02", "CIFAR_89", "Number_35"]

# The grid, in the order train_cnn.py iterates it. The data split is *not* a
# function of the grid position -- see fold_splits() -- so the order here only
# affects the sequence in which runs are executed.
GRID_DIMS = list(range(3, 8))
GRID_LAMBDAS = [1, 0]
GRID_FOLDS = [0, 1, 2, 3, 4]
NUM_TRAIN, NUM_TEST, VAL_RATIO = 200, 50, 0.2

# Backend defaults shared by the noisy-simulator and QPU stages
DEFAULT_BACKEND = "ibm_fez"
DEFAULT_SHOTS = 1024


def corr_list_for(lamb):
    """Cor values swept for this lambda; lambda=0 uses a single placeholder."""
    return [0.0, 0.5, 1.0] if lamb == 1 else [1]


def model_name_for(arch, seed):
    """Output directory name of one (arch, seed) run."""
    return f"{ARCH_CONFIG[arch]['model_name']}_seed{seed}"


def grid_tag(lambda_corr, correlation_strength, fold_idx):
    """Filename stem identifying one grid position."""
    return f"lamb{lambda_corr}_corr{correlation_strength}_fold{fold_idx}"


# ----------------------------------------------------------------------
# Model: CNN backbone + MLP classification head
# ----------------------------------------------------------------------
def build_backbone(arch: str, latent_dim: int, input_ch: int) -> nn.Sequential:
    """
    CNN backbone plus the shared projection head
    (AdaptiveAvgPool2d(4,1) -> Flatten -> Linear(d) -> BatchNorm1d(d)).

    "lightweight" : single conv block (32 channels).
    "complex"     : three conv blocks (32 -> 128 -> 512) with max pooling and a
                    dropout between blocks.
    """
    if arch == "lightweight":
        conv = [
            nn.Conv2d(input_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        ]
        flat_ch = 32
    elif arch == "complex":
        conv = [
            nn.Conv2d(input_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3),

            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        ]
        flat_ch = 512
    else:
        raise ValueError(f"Unknown arch: {arch!r}")

    return nn.Sequential(
        *conv,
        nn.Flatten(),
        nn.Linear(flat_ch * 4 * 1, latent_dim),
        nn.BatchNorm1d(latent_dim),   # BatchNorm before min-max scaling
    )


class CNN_MLP_Classifier(nn.Module):
    """
    CNN backbone (selected by `arch`) plus the MLP classification head.

    forward() returns (logits, z), where z is the d-dimensional feature vector
    that is later phase-encoded into the quantum feature map. The QSVM stages
    only load the state_dict, and dropout is inactive in eval mode.
    """

    def __init__(self, num_classes: int, arch: str, latent_dim: int = 32,
                 input_ch: int = 3, dropout: float = None):
        super().__init__()

        if dropout is None:
            dropout = ARCH_CONFIG[arch]["dropout"]

        self.cnn = build_backbone(arch, latent_dim, input_ch)

        # MLP classification head (its test accuracy is the "MLP" baseline)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)                # feature extraction -> z
        logits = self.classifier(x)    # classification logits
        return logits, x


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def balanced_take(idxs, labels, n, rng):
    """Draw n indices from `idxs`, class-balanced, using the given RNG."""
    labels = np.asarray(labels)
    idxs = list(idxs)

    y_sub = labels[idxs]
    classes = np.unique(y_sub)

    k = len(classes)
    base, rem = n // k, n % k

    chosen = []
    for i, c in enumerate(classes):
        c_idxs = [idxs[j] for j in np.where(y_sub == c)[0]]
        rng.shuffle(c_idxs)

        take = base + (1 if i < rem else 0)
        if take > len(c_idxs):
            raise ValueError(f"Not enough samples for class {c}: need {take}, have {len(c_idxs)}")
        chosen.extend(c_idxs[:take])

    rng.shuffle(chosen)
    return chosen


def fold_splits(y, seed, n_official_train):
    """
    Train/val/test indices of the 5 CV folds, for one (dataset, seed).

    The official protocol of MNIST / Fashion-MNIST / CIFAR-10 is respected:
    load_dataset() stacks the official train split before the official test
    split, so positions [0, n_official_train) are the official *train* pool and
    [n_official_train, len(y)) the official *test* pool. Training and validation
    samples are drawn only from the former, test samples only from the latter.
    The two never mix, at any fold or seed.

    Within each pool the 5 folds are cut with StratifiedKFold(shuffle, seed):
    a fold trains on the 4/5 majority part of the train pool and tests on its
    own 1/5 part of the test pool, so the five test sets are disjoint.

    The split is a function of (dataset, seed, fold) *only*. d, lambda and Cor
    are model hyper-parameters and must not move the data, so every grid
    position at the same fold trains, validates and tests on exactly the same
    samples -- that is what makes the Cor sweep a controlled comparison.

    Each fold gets its own RandomState(1000 * seed + fold), so the folds are
    also independent of each other and of the order they are visited in. No
    stage has to replay an RNG stream to know what a checkpoint was trained on;
    it just calls this function.

    Returns a list of 5 dicts (all indices are into the full stacked pool):
        train_idx : NUM_TRAIN balanced indices, official train split only
        test_idx  : NUM_TEST balanced indices, official test split only,
                    disjoint across folds
        val_pos   : positions *within train_idx* held out for validation
        train_pos : the remaining positions within train_idx
    """
    y = np.asarray(y)
    n_official_train = int(n_official_train)
    if not 0 < n_official_train < len(y):
        raise ValueError(f"n_official_train={n_official_train} out of range for {len(y)} samples")

    trainval_pool = np.arange(n_official_train)            # official train split
    test_pool = np.arange(n_official_train, len(y))        # official test split
    y_trainval, y_test = y[trainval_pool], y[test_pool]

    # Folded independently inside each pool; same seed, so fold k of the train
    # pool always pairs with fold k of the test pool.
    skf_trainval = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    skf_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    trainval_parts = [trainval_pool[part] for part, _ in
                      skf_trainval.split(np.zeros(len(y_trainval)), y_trainval)]
    test_parts = [test_pool[part] for _, part in
                  skf_test.split(np.zeros(len(y_test)), y_test)]

    splits = []
    for fold_idx, (trainval_all, test_all) in enumerate(zip(trainval_parts, test_parts)):
        rng = np.random.RandomState(1000 * seed + fold_idx)

        train_idx = balanced_take(trainval_all, y, NUM_TRAIN, rng)
        test_idx = balanced_take(test_all, y, NUM_TEST, rng)

        # Balanced validation split inside the training subset
        n_tr = len(train_idx)
        val_pos = balanced_take(range(n_tr), y[train_idx], int(n_tr * VAL_RATIO), rng)
        train_pos = sorted(set(range(n_tr)) - set(val_pos))

        splits.append(dict(train_idx=train_idx, test_idx=test_idx,
                           val_pos=val_pos, train_pos=train_pos))

    return splits


def filter_binary(dataset, class1, class2):
    """Keep the two classes (class1 -> 0, class2 -> 1) and unify to (N, C, H, W)."""
    targets = np.asarray(dataset.targets)
    mask = (targets == class1) | (targets == class2)

    X = dataset.data[mask]
    y = targets[mask]
    y = np.where(y == class1, 0, 1)

    if isinstance(X, torch.Tensor):
        X = X.numpy()

    if X.ndim == 3:
        # (N, H, W) -> (N, 1, H, W)
        X = X[:, None, :, :]
    elif X.ndim == 4:
        # (N, H, W, C) -> (N, C, H, W)
        if X.shape[-1] in (1, 3):
            X = np.transpose(X, (0, 3, 1, 2))
        elif X.shape[1] in (1, 3):
            pass
        else:
            raise ValueError(f"Unsupported 4D shape: {X.shape}")
    else:
        raise ValueError(f"Unsupported shape: {X.shape}")

    return X.astype(np.float32), y.astype(np.int64)


_dataset_cache = {}


def load_dataset(dataset_name, data_dir):
    """
    Binary (X, y) pool for dataset_name, plus the official train/test boundary.

    The official train and test splits are stacked in that order and never
    shuffled together, so the returned n_official_train marks the boundary:
    X[:n_official_train] is the official train split, X[n_official_train:] the
    official test split. fold_splits() draws from the two separately.

    Returns (X, y, classes, n_official_train).
    """
    if dataset_name in _dataset_cache:
        return _dataset_cache[dataset_name]

    classes = [int(dataset_name[-2]), int(dataset_name[-1])]

    transform = transforms.Compose([transforms.ToTensor()])
    if "CIFAR" in dataset_name:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    if "Fashion" in dataset_name:
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    if "Number" in dataset_name:
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    X_train, y_train = filter_binary(train_dataset, classes[0], classes[1])
    X_test, y_test = filter_binary(test_dataset, classes[0], classes[1])

    # Official train first, official test second -- the order is the protocol,
    # and n_official_train is the only thing that keeps them apart downstream.
    n_official_train = len(X_train)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    _dataset_cache[dataset_name] = (X, y, classes, n_official_train)
    return X, y, classes, n_official_train


# ----------------------------------------------------------------------
# Feature statistics
# ----------------------------------------------------------------------
def MAC(Z):
    """Mean absolute off-diagonal correlation of the feature matrix Z [n, d]."""
    Zc = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-8)
    corr = (Zc.T @ Zc) / (Zc.shape[0] - 1)
    d = corr.shape[0]
    if d > 1:
        mask = ~np.eye(d, dtype=bool)
        return float(np.mean(np.abs(corr[mask])))
    return 0.0


def same_diff_class_means(K: np.ndarray, y):
    """Mean kernel value over same-class pairs and different-class pairs."""
    y = np.asarray(y)
    same = (y[:, None] == y[None, :])
    diff = ~same
    return K[same].mean(), K[diff].mean()


def same_diff_class_vars(K: np.ndarray, y):
    """Variance of kernel values over same-class, different-class and all pairs."""
    y = np.asarray(y)
    same = (y[:, None] == y[None, :])
    diff = ~same
    return K[same].var(), K[diff].var(), K.var()


def each_class_means_vars(K: np.ndarray, y):
    """Per-class (0/0 and 1/1 pairs) mean and variance of kernel values."""
    y = np.asarray(y)
    class0 = (y[:, None] == 0) & (y[None, :] == 0)
    class1 = (y[:, None] == 1) & (y[None, :] == 1)
    c0, c1 = K[class0], K[class1]
    return c0.mean(), c1.mean(), c0.var(), c1.var()


# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------
def save_model(obj, save_dir, fold_idx, corr_lamb, correlation_strength):
    """Pickle a fitted (Q)SVC model."""
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, grid_tag(corr_lamb, correlation_strength, fold_idx) + ".pkl")
    joblib.dump(obj, fname)
    return str(fname)


def save_results_to_excel(fold_dir, model_type, kernel_or_fm, dims, lambda_corr, correlation_strength,
                          mac, fold, acc, f1, train_time, test_time):
    """Append one result row to <fold_dir>/result.xlsx (created on first call)."""
    filepath = os.path.join(fold_dir, "result.xlsx")
    if not os.path.exists(filepath):
        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(["Model", "Kernel/FeatureMap", "Dimension", "Lambda_corr", "Correlation_strength",
                   "MAC", "Fold", "Accuracy", "F1-score", "Train Time", "Test Time"])
        wb.save(filepath)

    wb = openpyxl.load_workbook(filepath)
    ws = wb["Results"]
    ws.append([model_type, kernel_or_fm, dims, lambda_corr, correlation_strength, mac, fold,
               acc, f1, train_time, test_time])
    wb.save(filepath)


def dump_kernel_stats_excel(excel_path, dims, lambda_corr, correlation_strength, K, labels):
    """Append kernel-statistics row (means/variances by class pairing) to an Excel file."""
    same_mean_q, diff_mean_q = same_diff_class_means(K, labels)
    class0_mean, class1_mean, class0_var, class1_var = each_class_means_vars(K, labels)
    same_var, diff_var, all_var = same_diff_class_vars(K, labels)

    print("==================")
    print(f"Dim: {dims}, Lamb: {lambda_corr}, Corr: {correlation_strength:.2f} | "
          f"Same: {same_mean_q:.4f} , Diff: {diff_mean_q:.4f}")
    print(f"Mean | Class0: {class0_mean:.4f} , Class1: {class1_mean:.4f}")
    print(f"Variance | Same: {same_var:.4f} , Diff: {diff_var:.4f}")
    print(f"Variance | Class0: {class0_var:.4f} , Class1: {class1_var:.4f}")
    print("==================")

    result_row = {
        "Dim": dims, "Lambda": lambda_corr, "Correlation": correlation_strength,
        "Same_mean_q": same_mean_q, "Diff_mean_q": diff_mean_q,
        "Class0_mean": class0_mean, "Class1_mean": class1_mean,
        "Same_var": same_var, "Diff_var": diff_var,
        "Class0_var": class0_var, "Class1_var": class1_var, "All_var": all_var,
    }
    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        df_new = pd.concat([df_old, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([result_row])
    df_new.to_excel(excel_path, index=False)


def save_portable_svm(qsvc, save_dir, fold_idx, corr_lamb, correlation_strength):
    """
    Dump a fitted QSVM's decision function as plain arrays, for run_qsvm_qpu.py.

    Unpickling the .pkl requires qiskit-machine-learning (environment A); the
    QPU stage runs in environment B and cannot load it. Everything that stage
    needs is

        f(x) = sum_i dual_coef_i * k(x, SV_i) + intercept

    so the support vectors (rows of the training matrix selected by support_),
    the dual coefficients, the intercept and the class labels are written to an
    .npz that any environment can read.
    """
    qpu_dir = os.path.join(save_dir, "qpu")
    os.makedirs(qpu_dir, exist_ok=True)
    fname = os.path.join(qpu_dir, grid_tag(corr_lamb, correlation_strength, fold_idx) + "_PORTABLE.npz")
    np.savez(fname,
             Xfit=qsvc._BaseLibSVM__Xfit,   # train_emb, in fit order
             support=qsvc.support_,         # indices of the support vectors
             dual_coef=qsvc.dual_coef_,
             intercept=qsvc.intercept_,
             classes=qsvc.classes_)
    return str(fname)
