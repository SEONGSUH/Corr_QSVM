"""
Noiseless simulator QSVM evaluation 

* Pipeline:
    1. Load the CNN checkpoints produced by training/train_cnn.py (same arch, same seed, same grid position) and extract d-dimensional features for the 200 train & 50 test samples.
    2. Normalize features: BatchNorm followed by min-max scaling to [0, pi].
    3. Phase-encode the features into a d-qubit feature map and evaluate the fidelity quantum kernel.
    4. Train and evaluate:
         - SVM-RBF baseline 
         - QSVM with ZFeatureMap (no entanglement)
         - QSVM with ZZFeatureMap (entanglement)
       All SVMs use scikit-learn defaults

* Environment with requirements_A.txt
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from common import (
    ARCH_CONFIG, DATASET_NAMES, DEFAULT_ARCHS, DEFAULT_DATA_DIR,
    DEFAULT_RESULT_ROOT, DEFAULT_SEEDS,
    GRID_DIMS, GRID_FOLDS, GRID_LAMBDAS, NUM_TRAIN, NUM_TEST, VAL_RATIO,
    MAC, CNN_MLP_Classifier, balanced_take, corr_list_for, dump_kernel_stats_excel,
    grid_tag, load_dataset, model_name_for, save_model, save_portable_svm,
    save_results_to_excel,
)

import warnings
warnings.filterwarnings("ignore")


def build_sampler(kind):
    if kind == "statevector":
        return StatevectorSampler()
    raise ValueError(f"Unknown sampler: {kind!r}")

def run_dataset(arch, seed_num, dataset_name, device, result_root, data_dir, sampler_kind):
    data_root = os.path.join(result_root, dataset_name)
    X, y, classes, n_official_train = load_dataset(dataset_name, data_dir)
    print(classes, data_root)

    # Paths: CNN checkpoints(input) and classifier outputs
    model_name = model_name_for(arch, seed_num)
    out_root = os.path.join(data_root, model_name) # CNN checkpoints
    classifier_root = os.path.join(data_root, model_name + "_Truncated")# outputs of this script
    os.makedirs(classifier_root, exist_ok=True)
    batch_size = 8   
    _splits = fold_splits(y, seed_num, n_official_train)

    # ZFeatureMap : single-qubit phase encoding, no entanglement
    # ZZFeatureMap: adds ZZ entangling phases
    feature_maps = [
        lambda n: ZFeatureMap(n, reps=1),
        lambda n: ZZFeatureMap(n, reps=1),
    ]
    fm_names = ["Z", "ZZ"]

    svm_acc, svm_f1 = [], []
    qsvm_acc, qsvm_f1 = [], []
    for dims in GRID_DIMS:
        # Saving circuit figures 
        for fm_name, fmap in zip(fm_names, feature_maps):
            fm = fmap(dims)
            circuit_dir = os.path.join(classifier_root, f"dim{dims}", "qsvm", fm_name)
            os.makedirs(circuit_dir, exist_ok=True)
            fm.decompose().draw("mpl", filename=os.path.join(circuit_dir, "circuit.png"))

        for lambda_corr in GRID_LAMBDAS:
            corr_list = corr_list_for(lambda_corr)

            for correlation_strength in corr_list:
                for fold_idx in GRID_FOLDS:
                    print(f"dims     : {dims}")
                    print(f"fold_idx : {fold_idx}")

                    split = _splits[fold_idx]
                    train_idx, test_idx = split["train_idx"], split["test_idx"]
                    val_idx, train_idx_inner = split["val_pos"], split["train_pos"]

                    X_tr, y_tr = X[train_idx, :, :, :], y[train_idx]
                    X_te, y_te = X[test_idx, :, :, :], y[test_idx]

                    train_ds = TensorDataset(torch.tensor(X_tr[train_idx_inner]), torch.tensor(y_tr[train_idx_inner]))
                    val_ds = TensorDataset(torch.tensor(X_tr[val_idx]), torch.tensor(y_tr[val_idx]))
                    test_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=batch_size)
                    test_loader = DataLoader(test_ds, batch_size=batch_size)

                    # Load the trained CNN and extract features
                    encoder_dir = os.path.join(out_root, f"dim{dims}",
                                               grid_tag(lambda_corr, correlation_strength, fold_idx) + ".pt")

                    model = CNN_MLP_Classifier(num_classes=len(np.unique(y_tr)), arch=arch,
                                               latent_dim=dims, input_ch=X_tr.shape[1]).to(device)
                    model.load_state_dict(torch.load(encoder_dir))

                    model.eval()
                    train_z, train_labels = [], []
                    test_z, test_labels = [], []
                    with torch.no_grad():
                        for xb, yb in train_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            _, z = model(xb)
                            train_z.append(z.cpu())
                            train_labels.extend(yb.cpu().numpy())
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            _, z = model(xb)
                            train_z.append(z.cpu())
                            train_labels.extend(yb.cpu().numpy())
                        for xb, yb in test_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            _, z = model(xb)
                            test_z.append(z.cpu())
                            test_labels.extend(yb.cpu().numpy())
                    train_emb = torch.cat(train_z, dim=0).numpy()
                    test_emb = torch.cat(test_z, dim=0).numpy()

                    # Feature normalization: min-max to [0, pi]
                    fig = plt.figure(figsize=(8, 4))
                    ax1 = plt.subplot(1, 2, 1)
                    ax1.hist(test_emb[:, 0])
                    ax1.set_title("Before min-max")

                    scaler = MinMaxScaler(feature_range=(0, np.pi))
                    train_emb = scaler.fit_transform(train_emb) # fir on training features
                    test_emb = scaler.transform(test_emb) #applied to thet test features

                    ax2 = plt.subplot(1, 2, 2)
                    ax2.hist(test_emb[:, 0])
                    ax2.set_title("After min-max")

                    plot_dir = os.path.join(classifier_root, f"dim{dims}", "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    plt.savefig(os.path.join(plot_dir, f"Feature0_dis_{grid_tag(lambda_corr, correlation_strength, fold_idx)}.png"))
                    plt.close()

                    # Correlation/ covariance heatmaps of the scaled test features
                    n = test_emb.shape[0]
                    mean = test_emb.mean(0, keepdims=True)
                    Zc = test_emb - mean
                    cov = (Zc.T @ Zc) / (n - 1)
                    std = test_emb.std(0, keepdims=True) + 1e-8
                    corr = cov / (std.T @ std)

                    plt.figure(figsize=(6, 5))
                    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f",
                                square=True, cbar=False, xticklabels=False, yticklabels=False)
                    plt.title(f"Corr matrix λ={lambda_corr}, s={correlation_strength}, fold={fold_idx}")
                    plt.savefig(os.path.join(plot_dir, f"CORR_{grid_tag(lambda_corr, correlation_strength, fold_idx)}.png"))
                    plt.close()

                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cov, cmap="coolwarm", vmin=-np.max(cov), vmax=np.max(cov), annot=True, fmt=".2f",
                                square=True, cbar=False, xticklabels=False, yticklabels=False)
                    plt.title(f"Cov matrix λ={lambda_corr}, s={correlation_strength}, fold={fold_idx}")
                    plt.savefig(os.path.join(plot_dir, f"COV_{grid_tag(lambda_corr, correlation_strength, fold_idx)}.png"))
                    plt.close()


                    # SVM-RBF baseline (scikit-learn default)
                    svm = SVC(kernel="rbf", decision_function_shape="ovo")
                    st = time.time()
                    svm.fit(train_emb, train_labels)
                    svm_training_time = time.time() - st

                    st = time.time()
                    y_pred = svm.predict(test_emb)
                    svm_test_time = time.time() - st
                    svm_acc.append(accuracy_score(test_labels, y_pred))
                    svm_f1.append(f1_score(test_labels, y_pred, average="macro"))

                    save_model(svm, os.path.join(classifier_root, f"dim{dims}", "svm"),
                                fold_idx, lambda_corr, correlation_strength)
                    save_results_to_excel(os.path.join(classifier_root, f"dim{dims}"), "SVM", "rbf",
                                          dims, lambda_corr, correlation_strength, MAC(test_emb), fold_idx,
                                          svm_acc[-1], svm_f1[-1], svm_training_time, svm_test_time)
                    print(f"[SVM] fold={fold_idx}, kernel=rbf, dim={dims}, λ={lambda_corr}, corr={correlation_strength} "
                          f"=> acc={svm_acc[-1]:.4f}, f1={svm_f1[-1]:.4f}, "
                          f"train={svm_training_time:.3f}s, test={svm_test_time:.3f}s")

                    # QSVM
                    for fm_name, fm_class in zip(fm_names, feature_maps):
                        fm = fm_class(dims)   # Eq. (5)

                        sampler = build_sampler(sampler_kind)
                        fidelity = ComputeUncompute(sampler=sampler)
                        kernel = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)
                        qsvc = QSVC(quantum_kernel=kernel)

                        start_time = time.time()
                        qsvc.fit(train_emb, train_labels)
                        qsvm_training_time = time.time() - start_time

                        start_time = time.time()
                        y_pred = qsvc.predict(test_emb)
                        qsvm_test_time = time.time() - start_time

                        qsvm_acc.append(accuracy_score(test_labels, y_pred))
                        qsvm_f1.append(f1_score(test_labels, y_pred, average="macro"))

                        qsvm_dir = os.path.join(classifier_root, f"dim{dims}", "qsvm", fm_name)
                        save_model(qsvc, qsvm_dir, fold_idx, lambda_corr, correlation_strength)
                        save_portable_svm(qsvc, qsvm_dir, fold_idx, lambda_corr, correlation_strength)
                        save_results_to_excel(os.path.join(classifier_root, f"dim{dims}"), "QSVM", fm_name,
                                              dims, lambda_corr, correlation_strength, MAC(test_emb), fold_idx,
                                              qsvm_acc[-1], qsvm_f1[-1], qsvm_training_time, qsvm_test_time)
                        print(f"[QSVM] fold={fold_idx}, fm={fm_name}, dim={dims}, λ={lambda_corr}, corr={correlation_strength} "
                              f"=> acc={qsvm_acc[-1]:.4f}, f1={qsvm_f1[-1]:.4f}, "
                              f"train={qsvm_training_time:.3f}s, test={qsvm_test_time:.3f}s")

                        # Kernel matrix statistics 
                        K_train = qsvc.quantum_kernel.evaluate(train_emb)
                        dump_kernel_stats_excel(
                            os.path.join(classifier_root, f"qsvm_{fm_name}_results_trainset.xlsx"),
                            dims, lambda_corr, correlation_strength, K_train, train_labels)

                        K_test = qsvc.quantum_kernel.evaluate(test_emb)
                        dump_kernel_stats_excel(
                            os.path.join(classifier_root, f"qsvm_{fm_name}_results_testset.xlsx"),
                            dims, lambda_corr, correlation_strength, K_test, test_labels)


# Parsing arguments
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Noiseless-simulator QSVM/SVM evaluation over the "
                    "arch x seed x dataset sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", nargs="+", choices=list(ARCH_CONFIG),
                        default=DEFAULT_ARCHS,
                        help="backbones to train")
    parser.add_argument("--seed", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="random seeds")
    parser.add_argument("--dataset", nargs="+", choices=DATASET_NAMES,
                        default=DATASET_NAMES,
                        help="binary datasets")
    parser.add_argument("--sampler", choices=["statevector"],
                        default="statevector",
                        help="sampler primitive.")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT,
                        help="directory for outputs")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="torchvision download cache")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    device = "cpu"  
    print(f"Using {device} device")
    print(f"Sweep: archs={args.arch}, seeds={args.seed}, datasets={args.dataset}\n")

    sampler_kind = args.sampler

    for arch in args.arch:
        for seed_num in args.seed:
            print("=" * 70)
            print(f"[SWEEP] arch={arch} ({ARCH_CONFIG[arch]['model_name']})  seed={seed_num}  "
                  f"sampler={sampler_kind}")
            print("=" * 70)
            for dataset_name in args.dataset:
                run_dataset(arch, seed_num, dataset_name, device,
                            args.result_root, args.data_dir, sampler_kind)


if __name__ == "__main__":
    main()
