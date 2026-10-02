"""
Noisy-simulator QSVM evaluation 

* Pipeline: Same as run_qsvm_noiseless.py except for the process that fidelity kernel [Eq. (6)] is estimated with a noisy simulator that emulates the ibm_fez device (Heron r2).

* Environment with requirements_B.txt

"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.primitives import BackendSamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from common import (
    ARCH_CONFIG, DATASET_NAMES, DEFAULT_ARCHS, DEFAULT_BACKEND, DEFAULT_DATA_DIR,
    DEFAULT_RESULT_ROOT, DEFAULT_SEEDS, DEFAULT_SHOTS,
    GRID_DIMS, GRID_FOLDS, GRID_LAMBDAS, NUM_TRAIN, NUM_TEST,
    MAC, CNN_MLP_Classifier, corr_list_for, dump_kernel_stats_excel, fold_splits,
    grid_tag, load_dataset, model_name_for, save_model, save_results_to_excel,
)

import warnings
warnings.filterwarnings("ignore")

# Noisy simulator emulating ibm_fez/Heron r2
def build_noisy_simulator(backend_name=DEFAULT_BACKEND):
    """
    - Building the Aer simulator carrying the device noise model.

    *** This phae requires saved IBM Quantum credentials --> -- run save_ibm_account.py once.
    The account is only used to fetch the backend's calibration/noise model (token not consumed through this code).
    """
    service = QiskitRuntimeService()
    real_backend = service.backend(backend_name)
    print(f"Connected to: {real_backend.name}")
    noisy_sim = AerSimulator.from_backend(real_backend)

    noisy_sim.set_options(
        method="statevector",
        max_parallel_threads=0,        # use all CPU cores
        max_parallel_experiments=0,    # parallelize over circuits
        max_parallel_shots=1,
        statevector_parallel_threshold=14,
    )
    # Truncate unused qubits of the 156-qubit device from the simulation
    noisy_sim.set_options(enable_truncation=True)

    sampler = Sampler(backend=noisy_sim)
    print(sampler.options)
    return noisy_sim, sampler


def run_dataset(arch, seed_num, dataset_name, device, result_root, data_dir,
                noisy_sim, sampler, num_shots):
    data_root = os.path.join(result_root, dataset_name)
    X, y, classes, n_official_train = load_dataset(dataset_name, data_dir)
    print(classes, data_root)

    # Paths: CNN checkpoints(input) and classifier outputs    
    model_name = model_name_for(arch, seed_num)
    out_root = os.path.join(data_root, model_name) # CNN checkpoints
    classifier_root = os.path.join(data_root, model_name + "_Truncated_NoisySim")# outputs of this script
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
        for lambda_corr in GRID_LAMBDAS:
            corr_list = corr_list_for(lambda_corr)

            for correlation_strength in corr_list:
                for fold_idx in GRID_FOLDS:
                    print(dataset_name)
                    print(f"dims     : {dims}")
                    print(f"lambda   : {lambda_corr}")
                    print(f"Cor      : {correlation_strength}")
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

                    # Load trained CNN and extract features
                    encoder_dir = os.path.join(out_root, f"dim{dims}",
                                               grid_tag(lambda_corr, correlation_strength, fold_idx) + ".pt")

                    model = CNN_MLP_Classifier(num_classes=len(np.unique(y_tr)), arch=arch,
                                               latent_dim=dims, input_ch=X_tr.shape[1]).to(device)
                    model.load_state_dict(torch.load(encoder_dir, map_location=torch.device('cpu')))

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
                    train_emb = scaler.fit_transform(train_emb)
                    test_emb = scaler.transform(test_emb)

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

                    # SVM-RBF baseline (scikit-learn defaults: C = 1.0, gamma='scale')
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


                    # Noisy QSVM
                    for fm_name, fm_class in zip(fm_names, feature_maps):
                        # ISA transpilation for the emulated device
                        pm = generate_preset_pass_manager(optimization_level=1, backend=noisy_sim,
                                                          initial_layout=list(range(dims)))
                        
                        num_samples = np.shape(train_emb)[0]
                        # Diagonal entries k(x, x) are 1 by definition (not measured)
                        kernel_matrix = np.eye(NUM_TRAIN)
                        test_matrix = np.full((NUM_TEST, num_samples), np.nan)

                        fm = fm_class(dims) 
                        circuit_dir = os.path.join(classifier_root, f"dim{dims}", "qsvm", fm_name)
                        os.makedirs(circuit_dir, exist_ok=True)
                        fm.decompose().draw("mpl", filename=os.path.join(circuit_dir, "ISA_circuit.png"))

                        # Training kernel all unordered pairs
                        all_circuits = []
                        pair_indices = []
                        for x1 in tqdm(range(NUM_TRAIN), desc="Training", leave=False):
                            for x2 in range(x1 + 1, NUM_TRAIN):
                                u1 = fm.assign_parameters(list(train_emb[x1]))
                                u2 = fm.assign_parameters(list(train_emb[x2]))
                                # Compute-uncompute overlap circuit
                                overlap_circ = u1.compose(u2.inverse())
                                overlap_circ.measure_all()

                                overlap_trans = pm.run(overlap_circ)
                                all_circuits.append(overlap_trans)
                                pair_indices.append((x1, x2))

                        qsvm_train_time = time.time()
                        job = sampler.run(all_circuits, shots=num_shots)
                        train_job_result = job.result()
                        breakpoint()
                        zero_state = "0" * dims
                        for idx, (x1, x2) in enumerate(pair_indices):
                            counts = train_job_result[idx].data.meas.get_counts()
                            prob = counts.get(zero_state, 0) / num_shots # empirical fidelity
                            kernel_matrix[x1, x2] = prob
                            kernel_matrix[x2, x1] = prob

                        # Precomputed-kernel SVM
                        qsvc = SVC(kernel="precomputed")
                        qsvc.fit(kernel_matrix, train_labels)
                        qsvm_train_time = time.time() - qsvm_train_time

                        # test kernel
                        all_test_circuits = []
                        test_pair_indices = []
                        print(f"Building test-matrix circuits ({NUM_TEST} x {NUM_TRAIN} = {NUM_TEST * NUM_TRAIN})...")
                        for x1 in tqdm(range(NUM_TEST), desc="Test", leave=False):
                            for x2 in range(NUM_TRAIN):
                                u_test = fm.assign_parameters(list(test_emb[x1]))
                                u_train = fm.assign_parameters(list(train_emb[x2]))

                                overlap_circ = u_test.compose(u_train.inverse())
                                overlap_circ.measure_all()

                                overlap_trans = pm.run(overlap_circ)
                                all_test_circuits.append(overlap_trans)
                                test_pair_indices.append((x1, x2))

                        qsvm_test_time = time.time()
                        job = sampler.run(all_test_circuits, shots=num_shots)
                        test_job_result = job.result()

                        zero_state = "0" * dims
                        for idx, (x1, x2) in enumerate(test_pair_indices):
                            counts = test_job_result[idx].data.meas.get_counts()
                            prob = counts.get(zero_state, 0) / num_shots
                            test_matrix[x1, x2] = prob

                        print("Test matrix complete.")

                        y_pred = qsvc.predict(test_matrix)
                        qsvm_test_time = time.time() - qsvm_test_time

                        qsvm_acc.append(accuracy_score(test_labels, y_pred))
                        qsvm_f1.append(f1_score(test_labels, y_pred, average="macro"))

                        save_model(qsvc, os.path.join(classifier_root, f"dim{dims}", "qsvm", fm_name),
                                    fold_idx, lambda_corr, correlation_strength)
                        save_results_to_excel(os.path.join(classifier_root, f"dim{dims}"), "QSVM", fm_name,
                                              dims, lambda_corr, correlation_strength, MAC(test_emb), fold_idx,
                                              qsvm_acc[-1], qsvm_f1[-1], qsvm_train_time, qsvm_test_time)
                        print(f"[QSVM] fold={fold_idx}, fm={fm_name}, dim={dims}, λ={lambda_corr}, corr={correlation_strength} "
                              f"=> acc={qsvm_acc[-1]:.4f}, f1={qsvm_f1[-1]:.4f}, "
                              f"train={qsvm_train_time:.3f}s, test={qsvm_test_time:.3f}s")

                        # Kernel-matrix statistics (training-set pairs)
                        dump_kernel_stats_excel(
                            os.path.join(classifier_root, f"qsvm_{fm_name}_results_trainset.xlsx"),
                            dims, lambda_corr, correlation_strength, kernel_matrix, train_labels)



# Command-line interface
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Noisy-simulator QSVM/SVM evaluation over the "
                    "arch x seed x dataset sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", nargs="+", choices=list(ARCH_CONFIG),
                        default=DEFAULT_ARCHS,
                        help="backbone(s) whose checkpoints are evaluated")
    parser.add_argument("--seed", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="seed(s); must match the CNN training runs")
    parser.add_argument("--dataset", nargs="+", choices=DATASET_NAMES,
                        default=DATASET_NAMES,
                        help="binary dataset code(s) to evaluate")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="IBM backend whose noise model is emulated")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                        help="shots per overlap circuit")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT,
                        help="root directory holding the CNN checkpoints and receiving outputs")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="torchvision download cache")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    device = "cpu" 
    print(f"Using {device} device")
    print(f"Sweep: archs={args.arch}, seeds={args.seed}, datasets={args.dataset}\n")

    noisy_sim, sampler = build_noisy_simulator(args.backend)

    for arch in args.arch:
        for seed_num in args.seed:
            print("=" * 70)
            print(f"[SWEEP] arch={arch} ({ARCH_CONFIG[arch]['model_name']})  seed={seed_num}")
            print("=" * 70)
            for dataset_name in args.dataset:
                run_dataset(arch, seed_num, dataset_name, device,
                            args.result_root, args.data_dir,
                            noisy_sim, sampler, args.shots)


if __name__ == "__main__":
    main()
