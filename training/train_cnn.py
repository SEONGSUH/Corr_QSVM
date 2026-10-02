"""
CNN feature-extractor training for lightweight/compex cnn


- Experimental grid per run: 2 backbones & 3 seeds & 5 binary datasets & d in {3..7} & Cor in {0.0, 0.5, 1.0} & 5 folds.

- Outputs per (arch, seed, dataset, d, lambda, Cor, fold):
    - checkpoint: <RESULT_ROOT>/<dataset>/<MODEL_NAME>_seed<SEED>/dim<d>/
                  lamb{lambda}_corr{Cor}_fold{fold}.pt
    - training curves and correlation/covariance heatmaps under .../plots/
    - summary.xlsx: test accuracy of the CNN+MLP classifier (this is the
      "MLP" baseline reported in the paper), per-class F1, and MAC
      (mean absolute off-diagonal correlation) of the test features.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from common import (
    ARCH_CONFIG, DATASET_NAMES, DEFAULT_ARCHS, DEFAULT_DATA_DIR,
    DEFAULT_RESULT_ROOT, DEFAULT_SEEDS,
    GRID_DIMS, GRID_FOLDS, GRID_LAMBDAS,
    CNN_MLP_Classifier, corr_list_for, fold_splits, grid_tag, load_dataset,
    model_name_for,
)

import warnings
warnings.filterwarnings("ignore")

def add_correlation_loss(latent, strength=0.5): # Correlation loss 
    """
    L_corr = (1/d^2) * || C - sigma(Cor) ||_F^2 

    C          : empirical correlation matrix of the batch-standardized
                 features (latent standardized per feature within the batch).
    sigma(Cor) : target matrix with ones on the diagonal and the uniform
                 off-diagonal value Cor = `strength`.
    The (1/d^2) factor is realized by torch.mean over the d x d entries.
    """
    latent = latent - latent.mean(dim=0, keepdim=True)
    latent = latent / (latent.std(dim=0, keepdim=True) + 1e-8)
    batch_size = latent.size(0)
    corr_matrix = (latent.T @ latent) / (batch_size - 1)

    identity = torch.eye(latent.shape[1], device=latent.device)
    ones = torch.ones_like(corr_matrix)

    # sigma(Cor): diagonal = 1, off-diagonal = strength
    target = strength * ones + (1 - strength) * identity

    loss = torch.mean((corr_matrix - target) ** 2)
    return loss


def run_dataset(arch, seed_num, dataset_name, device, result_root, data_dir): # 

    data_root = os.path.join(result_root, dataset_name) 
    X, y, classes, n_official_train = load_dataset(dataset_name, data_dir)
    print(classes, data_root)

    # Training setup
    out_root = os.path.join(data_root, model_name_for(arch, seed_num))
    os.makedirs(out_root, exist_ok=True)

    epochs, batch_size, patience, lr = 50, 8, 10, 1e-3

    
    # 5-fold CV splits, a function of (dataset, seed, fold); train/val come from
    # the official train split, test from the official test split.
    _splits = fold_splits(y, seed_num, n_official_train)

    results = []

    for dims in GRID_DIMS:
        for lambda_corr in GRID_LAMBDAS:
            corr_list = corr_list_for(lambda_corr)

            for correlation_strength in corr_list:
                for fold_idx in GRID_FOLDS:
                    print(f"dims     : {dims}")
                    print(f"fold_idx : {fold_idx}")

                    # Balanced 200 train (incl. 20% val) / 50 test of this fold
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

                    # Model /optimizer
                    model = CNN_MLP_Classifier(
                        num_classes=len(np.unique(y_tr)), arch=arch,
                        latent_dim=dims, input_ch=X_tr.shape[1]).to(device)
                    optim = Adam(model.parameters(), lr=lr)
                    ce = nn.CrossEntropyLoss()

                    best_val, bad, best_ep = -1, 0, -1
                    tr_losses, va_losses, tr_accs, va_accs = [], [], [], []

                    fold_dir = os.path.join(out_root, f"dim{dims}")
                    os.makedirs(fold_dir, exist_ok=True)
                    model_dir = os.path.join(fold_dir, grid_tag(lambda_corr, correlation_strength, fold_idx) + ".pt")
                 
                    # Training loop
                    for ep in range(1, epochs + 1):
                        model.train()
                        run_loss = run_acc = n = 0
                        pbar = tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {ep}/{epochs} [Train]", leave=False)
                        for xb, yb in pbar:
                            xb, yb = xb.to(device), yb.to(device)
                            optim.zero_grad()
                            logits, z = model(xb)
                            loss_cls = ce(logits, yb)
                            loss_corr = add_correlation_loss(z, correlation_strength)  # Eq. (10)
                            loss = loss_cls + lambda_corr * loss_corr  # Total loss: unweighted sum  L=L_cls+lambda*L_corr

                            loss.backward()
                            optim.step()
                            preds = logits.argmax(1)
                            acc = (preds == yb).float().mean().item()
                            bs = yb.size(0)
                            run_loss += loss.item() * bs
                            run_acc += acc * bs
                            n += bs
                            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
                        tr_loss = run_loss / n
                        tr_acc = run_acc / n


                        # Validation phase
                        model.eval()
                        run_loss = run_acc = n = 0
                        pbar = tqdm(val_loader, desc=f"[Fold {fold_idx}] Epoch {ep}/{epochs} [Val]", leave=False)
                        with torch.no_grad():
                            for xb, yb in pbar:
                                xb, yb = xb.to(device), yb.to(device)
                                logits, z = model(xb)
                                loss = ce(logits, yb)
                                preds = logits.argmax(1)
                                acc = (preds == yb).float().mean().item()
                                bs = yb.size(0)
                                run_loss += loss.item() * bs
                                run_acc += acc * bs
                                n += bs
                                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
                        va_loss = run_loss / n
                        va_acc = run_acc / n

                        print(f"[Fold {fold_idx}] Epoch {ep}/{epochs} | "
                              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | Val Loss {va_loss:.4f} Acc {va_acc:.4f}")

                        tr_losses.append(tr_loss)
                        va_losses.append(va_loss)
                        tr_accs.append(tr_acc)
                        va_accs.append(va_acc)

                        if va_acc > best_val:
                            best_val, best_ep, bad = va_acc, ep, 0
                            torch.save(model.state_dict(), model_dir)
                        else:
                            bad += 1
                        if bad >= patience:
                            print(f"[Fold {fold_idx}] Early stop at epoch {ep}")
                            break

                    print(f"[Fold {fold_idx}] best_va_acc={best_val:.4f}")
                    plot_dir = os.path.join(fold_dir, "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    plt.figure()
                    plt.plot(tr_accs, label="Train")
                    plt.plot(va_accs, label="Val")
                    plt.legend()
                    plt.title("Acc")
                    plt.savefig(os.path.join(plot_dir, f"ACC_{grid_tag(lambda_corr, correlation_strength, fold_idx)}.png"))
                    plt.close()

                    # Test and feature correlation statistics
                    model.load_state_dict(torch.load(model_dir))
                    model.eval()
                    all_z = []
                    run_acc = n = 0
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for xb, yb in test_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            logits, z = model(xb)
                            preds = logits.argmax(1)
                            acc = (preds == yb).float().mean().item()
                            run_acc += acc * yb.size(0)
                            n += yb.size(0)
                            all_z.append(z.cpu())
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(yb.cpu().numpy())
                    test_acc = run_acc / n           
                    f1_per_class = f1_score(all_labels, all_preds, average=None)

                    Z = torch.cat(all_z, dim=0)# [n, d]
                    n = Z.shape[0]

                    mean = Z.mean(dim=0, keepdim=True)
                    Zc = Z - mean
                    cov = (Zc.T @ Zc) / (n - 1)

                    std = Z.std(dim=0, keepdim=True, unbiased=True)
                    corr = cov / (std.T @ std).clamp_min(1e-12)

                    d = corr.shape[0]
                    if d > 1:
                        mask = ~torch.eye(d, dtype=torch.bool, device=corr.device)
                        mean_abs_corr = corr[mask].abs().mean().item()
                    else:
                        mean_abs_corr = 0.0

                    plt.figure(figsize=(6, 5))
                    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f",
                                square=True, cbar=False, xticklabels=False, yticklabels=False)
                    plt.title(f"Corr matrix λ={lambda_corr}, s={correlation_strength}, fold={fold_idx}")
                    plt.savefig(os.path.join(plot_dir, f"CORR_{grid_tag(lambda_corr, correlation_strength, fold_idx)}.png"))
                    plt.close()

                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cov, cmap="coolwarm", vmin=-torch.max(cov), vmax=torch.max(cov), annot=True, fmt=".2f",
                                square=True, cbar=False, xticklabels=False, yticklabels=False)
                    plt.title(f"Cov matrix λ={lambda_corr}, s={correlation_strength}, fold={fold_idx}")
                    plt.savefig(os.path.join(plot_dir, f"COV_{grid_tag(lambda_corr, correlation_strength, fold_idx)}.png"))
                    plt.close()

                    results.append(dict(
                        dims=dims,
                        lambda_corr=lambda_corr,
                        strength=correlation_strength,
                        fold=fold_idx,
                        best_val=best_val,
                        best_ep=best_ep,
                        test_acc=test_acc,
                        mean_abs_corr=mean_abs_corr,
                        **{f"f1_class{i}": f1 for i, f1 in enumerate(f1_per_class)}
                    ))
                    print(results[-1])

    # Save summary (per arch, seed, dataset)
    df = pd.DataFrame(results)
    df.to_excel(os.path.join(out_root, "summary.xlsx"), index=False)
    print("[DONE] summary.xlsx saved")


# Arguments parsing
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the CNN feature extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", nargs="+", choices=list(ARCH_CONFIG),
                        default=DEFAULT_ARCHS,
                        help="backbones to train")
    parser.add_argument("--seed", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="random seeds")
    parser.add_argument("--dataset", nargs="+", choices=DATASET_NAMES,
                        default=DATASET_NAMES,
                        help="binary dataset")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT,
                        help="directory for results")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="torchvision download cache")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda when available, else cpu)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device} device")
    print(f"Sweep: archs={args.arch}, seeds={args.seed}, datasets={args.dataset}\n")
    
    for arch in args.arch:
        for seed_num in args.seed:
            print("=" * 70)
            print(f"[SWEEP] arch={arch} ({ARCH_CONFIG[arch]['model_name']})  seed={seed_num}")
            print("=" * 70)
            for dataset_name in args.dataset:
                run_dataset(arch, seed_num, dataset_name, device,
                            args.result_root, args.data_dir)


if __name__ == "__main__":
    main()
