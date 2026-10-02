"""
QPU for QSVM inference only

* Pipeline:
  The QSVM of that position was fitted in run_qsvm_noiseless.py; here only the
  test-time kernel column K(x_test, SV) is re-measured on the QPU and fed to the
  stored decision function, so no re-training happens.

* Environment with requirements_B.txt

* Credentials come from the saved IBM Quantum account -- run save_ibm_account.py.

Usage example:
    python training/run_qsvm_qpu.py run --dataset CIFAR_89 --arch complex --seed 2 --mode statevector
    python training/run_qsvm_qpu.py run --dataset CIFAR_89 --arch complex --seed 2 --mode qpu --rows 2
    python training/run_qsvm_qpu.py view --dataset CIFAR_89 --arch complex --seed 2
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from common import (
    ARCH_CONFIG, DATASET_NAMES, DEFAULT_BACKEND, DEFAULT_DATA_DIR,
    DEFAULT_RESULT_ROOT, DEFAULT_SHOTS,
    GRID_DIMS, GRID_FOLDS, GRID_LAMBDAS,
    CNN_MLP_Classifier, fold_splits, grid_tag, load_dataset,
    model_name_for,
)

DEFAULT_MAX_CIRCUITS = 300

import warnings
warnings.filterwarnings("ignore")


def normalize_cor(lamb, cor):
    """
    Restore the literal Cor value train_cnn.py used in its filenames.

    The lambda=0 branch loops over the int placeholder 1 and writes "corr1";
    the lambda=1 branch loops over floats and writes "corr0.0"/"corr0.5"/
    "corr1.0". argparse turns both into floats, which would make the lambda=0
    filenames come out as "corr1.0" and never match on disk.
    """
    if lamb == 0:
        if cor not in (1, 1.0):
            print(f"[warn] lambda=0 has a single placeholder Cor; ignoring --cor {cor}")
        return 1
    if cor not in (0.0, 0.5, 1.0):
        raise SystemExit(f"--cor must be 0.0, 0.5 or 1.0 when --lamb 1 (got {cor})")
    return float(cor)



def reproduce_indices(y, seed, fold, n_official_train):
    split = fold_splits(y, seed, n_official_train)[fold]
    return split["train_idx"], split["test_idx"]


@torch.no_grad()
def extract_emb(model, X, idx, device, bs=256):
    out = []
    for i in range(0, len(idx), bs):
        xb = torch.tensor(X[idx[i:i + bs]]).to(device)
        out.append(model(xb)[1].cpu().numpy())
    return np.concatenate(out, 0)

# Paths
def target_paths(args):
    """Every path this script reads or writes for the selected grid position."""
    model_name = model_name_for(args.arch, args.seed)
    dataset_root = os.path.join(args.result_root, args.dataset)

    tag = grid_tag(args.lamb, args.cor, args.fold)
    ckpt_root = os.path.join(dataset_root, model_name)
    qsvm_root = os.path.join(dataset_root, model_name + "_Truncated",
                             f"dim{args.dim}", "qsvm", args.fm)
    qpu_root = os.path.join(qsvm_root, "qpu")

    return dict(
        tag=tag,
        encoder=os.path.join(ckpt_root, f"dim{args.dim}", f"{tag}.pt"),
        qsvc_pkl=os.path.join(qsvm_root, f"{tag}.pkl"),
        qpu_root=qpu_root,
        portable=os.path.join(qpu_root, f"{tag}_PORTABLE.npz"),
        kernel=os.path.join(qpu_root, f"{tag}_Kqpu.npz"),
        qpy=os.path.join(qpu_root, f"fm_{args.fm}_dim{args.dim}.qpy"),
    )

# Subcommand: export
def _load_fitted_model(path):
    #Read a QSVC pickle without importing the classes it was pickled with.

    import inspect

    import joblib
    from joblib.numpy_pickle import NumpyUnpickler

    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"[export] plain load failed ({type(exc).__name__}: {exc}); "
              f"retrying with placeholder classes")

    class _Placeholder:
        # __init__ defined so object.__new__ tolerates the pickled constructor args
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, tuple) and len(state) == 2:
                attrs, slots = state
            else:
                attrs, slots = state, None
            for part in (attrs, slots):
                if isinstance(part, dict):
                    self.__dict__.update(part)


        def append(self, value):
            pass

        def extend(self, values):
            pass

        def add(self, value):
            pass

        def __setitem__(self, key, value):
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            stub = _Placeholder()
            object.__setattr__(self, name, stub)
            return stub

    def _stub(module, name):
        return type(name, (_Placeholder,), {"__module__": module})

    class _PermissiveUnpickler(NumpyUnpickler):
        def find_class(self, module, name):
            if module.split(".")[0].startswith("qiskit"):
                return _stub(module, name)
            try:
                return super().find_class(module, name)
            except Exception:
                return _stub(module, name)

    params = inspect.signature(NumpyUnpickler.__init__).parameters
    extra = {}
    if "ensure_native_byte_order" in params:
        extra["ensure_native_byte_order"] = True

    with open(path, "rb") as fh:
        return _PermissiveUnpickler(path, fh, **extra).load()


def cmd_export(args):
    paths = target_paths(args)
    if not os.path.exists(paths["qsvc_pkl"]):
        raise SystemExit(f"fitted QSVM not found: {paths['qsvc_pkl']}\n"
                         f"Run run_qsvm_noiseless.py for this grid position first.")

    qsvc = _load_fitted_model(paths["qsvc_pkl"])

    fields = {"Xfit": "_BaseLibSVM__Xfit", "support": "support_",
              "dual_coef": "dual_coef_", "intercept": "intercept_",
              "classes": "classes_"}
    arrays = {}
    for key, attr in fields.items():
        value = getattr(qsvc, attr, None)
        if not isinstance(value, np.ndarray):
            raise SystemExit(f"{attr} did not come back as an array "
                             f"(got {type(value).__name__}); the pickle could not be "
                             f"read here -- re-run run_qsvm_noiseless.py instead, "
                             f"it writes this .npz directly.")
        arrays[key] = value

    os.makedirs(paths["qpu_root"], exist_ok=True)
    np.savez(paths["portable"], **arrays)

    print(f"[export] {paths['portable']}")
    print(f"         Xfit {arrays['Xfit'].shape}, {len(arrays['support'])} support vectors")


# Feature map and kernels
def load_feature_map(paths, args):
    """Prefer the QPY dumped next to the kernel; otherwise rebuild it."""
    from qiskit import qpy

    if os.path.exists(paths["qpy"]):
        with open(paths["qpy"], "rb") as f:
            fm = qpy.load(f)[0]
        print(f"[fm] loaded from {os.path.basename(paths['qpy'])}")
        return fm

    from qiskit.circuit.library import z_feature_map, zz_feature_map
    if args.fm == "ZZ":
        fm = zz_feature_map(feature_dimension=args.dim, reps=1, entanglement="full")
    else:
        fm = z_feature_map(feature_dimension=args.dim, reps=1)

    os.makedirs(paths["qpu_root"], exist_ok=True)
    with open(paths["qpy"], "wb") as f:
        qpy.dump(fm, f)
    print(f"[fm] rebuilt and saved to {os.path.basename(paths['qpy'])}")
    return fm


def _overlap_circuits(fm, A, B):
    """Compute-uncompute overlap circuits U(a) U(b)^dagger for every (a, b)."""
    circ, idx = [], []
    for i in range(len(A)):
        ua = fm.assign_parameters(list(A[i]))
        for j in range(len(B)):
            c = ua.compose(fm.assign_parameters(list(B[j])).inverse())
            c.measure_all()
            circ.append(c)
            idx.append((i, j))
    return circ, idx


def fidelity_exact(fm, A, B):
    """Exact |<phi(a)|phi(b)>|^2 from statevectors -- the noiseless reference."""
    from qiskit.quantum_info import Statevector

    svA = [Statevector(fm.assign_parameters(list(a))) for a in A]
    svB = [Statevector(fm.assign_parameters(list(b))) for b in B]
    return np.array([[abs(a.inner(b)) ** 2 for b in svB] for a in svA])


def fidelity_local(fm, A, B, shots):
    """Local sampler: same measure -> counts path as the QPU, no credentials."""
    from qiskit.primitives import StatevectorSampler

    sampler = StatevectorSampler()
    circ, idx = _overlap_circuits(fm, A, B)
    res = sampler.run(circ, shots=shots).result()

    zero = "0" * A.shape[1]
    K = np.full((len(A), len(B)), np.nan)
    for k, (i, j) in enumerate(idx):
        K[i, j] = res[k].data.meas.get_counts().get(zero, 0) / shots
    return K

# Resumable QPU kernel
def _load_or_init(path, n_test, n_sv, meta):
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        K = d["K"]
        saved_meta = json.loads(str(d["meta"]))
        if saved_meta != meta:
            raise SystemExit(
                "meta mismatch -- this kernel file belongs to a different target/backend/setting.\n"
                f"  saved  : {saved_meta}\n  current: {meta}\n"
                "  Delete the file to start a fresh accumulation.")
        if K.shape != (n_test, n_sv):
            raise SystemExit(f"shape mismatch {K.shape} != {(n_test, n_sv)}")
        done = int((~np.isnan(K).any(axis=1)).sum())
        print(f"[resume] {path}\n         {done}/{n_test} rows done, {n_test - done} left")
        return K

    print(f"[new] {path}  ({n_test} x {n_sv})")
    return np.full((n_test, n_sv), np.nan)


def _save(path, K, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"          # .npz suffix so np.savez does not append one
    np.savez(tmp, K=K, meta=json.dumps(meta))
    os.replace(tmp, path)            # atomic swap: a crash cannot corrupt the file


def _log_job(path, info):
    with open(path + ".jobs.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")


def run_resumable_qpu(fm, test_emb, SV, kmat_path, n_rows,
                      shots=DEFAULT_SHOTS, max_circuits=DEFAULT_MAX_CIRCUITS,
                      opt_level=1, backend_name=DEFAULT_BACKEND):

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    n_test, n_sv = len(test_emb), len(SV)
    dims = SV.shape[1]
    zero = "0" * dims
    meta = {"tag": os.path.basename(kmat_path), "n_sv": int(n_sv), "dims": int(dims),
            "shots": int(shots), "fm_qubits": int(fm.num_qubits), "backend": backend_name}

    K = _load_or_init(kmat_path, n_test, n_sv, meta)

    pending = np.where(np.isnan(K).any(axis=1))[0]
    if len(pending) == 0:
        print("[done] every row finished; nothing to submit.")
        return K

    todo = [int(i) for i in pending[:n_rows]]
    print(f"[plan] this run: rows {todo}  ({len(todo)} rows, {len(todo) * n_sv} circuits)")

    service = QiskitRuntimeService()          # saved account; see save_ibm_account.py
    backend = service.backend(backend_name)
    try:
        pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    except Exception:                          # transpiler plugin conflicts
        pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend,
                                          translation_method="translator")
    print(f"[backend] {backend.name}")

    circ, idx = [], []
    for i in todo:
        ua = fm.assign_parameters(list(test_emb[i]))
        for j in range(n_sv):
            c = ua.compose(fm.assign_parameters(list(SV[j])).inverse())
            c.measure_all()
            circ.append(pm.run(c))
            idx.append((i, j))

    sampler = Sampler(mode=backend)
    for st in range(0, len(circ), max_circuits):
        sub = circ[st:st + max_circuits]
        job = sampler.run(sub, shots=shots)
        jid = job.job_id()
        print(f"  [job] {jid}  circuits {st}~{st + len(sub) - 1} submitted, waiting...")
        res = job.result()
        for k, pub in enumerate(res):
            i, j = idx[st + k]
            K[i, j] = pub.data.meas.get_counts().get(zero, 0) / shots
        _save(kmat_path, K, meta)
        _log_job(kmat_path, {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "job_id": jid, "backend": backend.name,
                             "rows": todo, "n_circuits": len(sub), "shots": shots})
        print(f"  saved ({st + len(sub)}/{len(circ)} circuits)")

    done = int((~np.isnan(K).any(axis=1)).sum())
    print(f"[ok] {done}/{n_test} rows accumulated -> {kmat_path}")
    return K

# Decision function
def predict_from_K(K, portable_path, y_test, label=""):
    # Apply the stored SVM decision function; score only the finished rows.
    d = np.load(portable_path)
    dual = d["dual_coef"][0]
    b = d["intercept"][0]
    cls = d["classes"]

    full = ~np.isnan(K).any(axis=1)
    if not full.all():
        print(f"[partial] {full.sum()}/{len(K)} rows finished -- scoring those only")

    Kc, yc = K[full], np.asarray(y_test)[full]
    if len(yc) == 0:
        print("[eval] no finished rows to score")
        return None, None, None

    pred = np.where(Kc @ dual + b > 0, cls[1], cls[0])
    acc = accuracy_score(yc, pred)
    f1 = f1_score(yc, pred, average="macro")
    print(f"[{label or 'eval'}] n={len(yc)}  acc={acc:.4f}  f1={f1:.4f}")
    return pred, acc, f1


# Subcommand: run  (environment B)
def cmd_run(args):
    paths = target_paths(args)
    print(f"=== target: {args.dataset} {args.arch} seed{args.seed} dim{args.dim} "
          f"{paths['tag']} fm={args.fm}  (mode={args.mode}) ===")

    for key, path in [("encoder", paths["encoder"]), ("portable", paths["portable"])]:
        if not os.path.exists(path):
            hint = ("  Run train_cnn.py for this grid position first." if key == "encoder"
                    else "  run_qsvm_noiseless.py writes this file; for a .pkl fitted "
                         "before that, use the 'export' subcommand.")
            raise SystemExit(f"{key} not found: {path}\n{hint}")

    device = torch.device("cpu")# feature extraction only

    # data and the exact subsets this grid position was trained on
    X, y, classes, n_official_train = load_dataset(args.dataset, args.data_dir)
    train_idx, test_idx = reproduce_indices(y, args.seed, args.fold, n_official_train)

    model = CNN_MLP_Classifier(num_classes=len(np.unique(y)), arch=args.arch,
                               latent_dim=args.dim, input_ch=X.shape[1]).to(device)
    model.load_state_dict(torch.load(paths["encoder"], map_location=device))
    model.eval()

    raw_train = extract_emb(model, X, train_idx, device)
    raw_test = extract_emb(model, X, test_idx, device)
    y_test = y[test_idx]

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaled_train = scaler.fit_transform(raw_train)
    test_emb = scaler.transform(raw_test)

    #stored decision function
    d = np.load(paths["portable"])
    Xfit = d["Xfit"]
    SV = Xfit[d["support"]]
    dual = d["dual_coef"][0]
    b = d["intercept"][0]
    cls = d["classes"]
    print(f"[info] {len(SV)} support vectors, {len(test_emb)} test samples")

    D = cdist(scaled_train, Xfit)
    gap = max(D.min(axis=1).max(), D.min(axis=0).max())
    print(f"[check] nearest-neighbour distance = {gap:.2e} -> {'OK' if gap < 1e-3 else 'MISMATCH'}")
    if gap >= 1e-3:
        raise SystemExit("reproduced train features do not match Xfit -- "
                         "check --dim/--lamb/--cor/--fold/--seed and the checkpoint")

    fm = load_feature_map(paths, args)

    # noiseless reference for the whole pipelin
    if not args.skip_exact:
        K0 = fidelity_exact(fm, test_emb, SV)
        p0 = np.where(K0 @ dual + b > 0, cls[1], cls[0])
        print(f"[noiseless] n={len(y_test)}  acc={accuracy_score(y_test, p0):.4f}  "
              f"f1={f1_score(y_test, p0, average='macro'):.4f}")

    # measured kernel
    if args.mode == "statevector":
        te_run = test_emb[:args.rows] if args.rows else test_emb
        K = fidelity_local(fm, te_run, SV, shots=args.shots)
        predict_from_K(K, paths["portable"], y_test[:len(te_run)], label="statevector")
    else:
        K = run_resumable_qpu(fm, test_emb, SV, paths["kernel"],
                              n_rows=args.rows, shots=args.shots,
                              max_circuits=args.max_circuits, backend_name=args.backend)
        predict_from_K(K, paths["portable"], y_test, label="qpu")


# Subcommand, view
def cmd_view(args):
    paths = target_paths(args)
    if not os.path.exists(paths["kernel"]):
        raise SystemExit(f"no accumulated kernel yet: {paths['kernel']}")

    d_k = np.load(paths["kernel"], allow_pickle=True)
    K = d_k["K"]
    meta = json.loads(str(d_k["meta"])) if "meta" in d_k.files else {}

    done_mask = ~np.isnan(K).any(axis=1)
    todo_rows = np.where(~done_mask)[0]

    print(f"=== {paths['tag']} ===")
    print(f"meta       : {meta}")
    print(f"K shape    : {K.shape}")
    print(f"rows done  : {done_mask.sum()}/{len(K)}   pending: {list(todo_rows)}")
    if done_mask.any():
        print(f"fidelity   : [{np.nanmin(K):.3f}, {np.nanmax(K):.3f}]")

    # y_test is re-derived from the seed, so no encoder or credentials needed
    _, y, _, n_official_train = load_dataset(args.dataset, args.data_dir)
    _, test_idx = reproduce_indices(y, args.seed, args.fold, n_official_train)
    y_test = y[test_idx]

    pred, _, _ = predict_from_K(K, paths["portable"], y_test, label="eval")
    if pred is not None:
        done_rows = np.where(done_mask)[0]
        wrong = np.where(pred != np.asarray(y_test)[done_mask])[0]
        if len(wrong):
            print(f"       misclassified test rows: {list(done_rows[wrong])}")

    jobs_path = paths["kernel"] + ".jobs.txt"
    if os.path.exists(jobs_path):
        with open(jobs_path, encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
        print(f"\n[jobs] {len(logs)} jobs, {sum(j['n_circuits'] for j in logs)} circuits total")
        for j in logs[-10:]:
            print(f"  {j['time']}  {j['job_id']}  {j['backend']}  "
                  f"circuits={j['n_circuits']} shots={j['shots']}")
    else:
        print("\n(no job log)")


# Command-line interface
def add_target_args(p):
    """Arguments identifying one grid position; shared by all subcommands."""
    p.add_argument("--dataset", choices=DATASET_NAMES, required=True, help="binary dataset code")
    p.add_argument("--arch", choices=list(ARCH_CONFIG), required=True,help="CNN backbone of the checkpoint")
    p.add_argument("--seed", type=int, required=True,help="seed of the training run")
    p.add_argument("--dim", type=int, choices=GRID_DIMS, default=3,help="feature dimension d (= number of qubits)")
    p.add_argument("--lamb", type=int, choices=GRID_LAMBDAS, default=1,help="lambda of the correlation loss")
    p.add_argument("--cor", type=float, default=0.0,help="Cor value; use 1 when --lamb 0")
    p.add_argument("--fold", type=int, choices=GRID_FOLDS, default=0)
    p.add_argument("--fm", choices=["Z", "ZZ"], default="Z",help="feature map of the fitted QSVM")
    p.add_argument("--result-root", default=DEFAULT_RESULT_ROOT,help="root directory holding the checkpoints and QSVMs")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,help="torchvision download cache")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="QPU inference for one trained QSVM grid position.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_exp = sub.add_parser("export",help="fallback: dump a fitted QSVC to a portable .npz "
                                "(run_qsvm_noiseless.py normally writes it)",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_target_args(p_exp)
    p_exp.set_defaults(func=cmd_export)

    p_run = sub.add_parser("run", help="measure the kernel locally or on the QPU",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_target_args(p_run)
    p_run.add_argument("--mode", choices=["statevector", "qpu"], default="statevector",help="'statevector' runs locally without credentials")
    p_run.add_argument("--rows", type=int, default=2,help="test rows to measure this run (0 = all, statevector only)")
    p_run.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p_run.add_argument("--max-circuits", type=int, default=DEFAULT_MAX_CIRCUITS,help="circuits per job")
    p_run.add_argument("--backend", default=DEFAULT_BACKEND,help="IBM backend to run on")
    p_run.add_argument("--skip-exact", action="store_true", help="skip the noiseless statevector reference")
    p_run.set_defaults(func=cmd_run)

    p_view = sub.add_parser("view", help="inspect an accumulated kernel file",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_target_args(p_view)
    p_view.set_defaults(func=cmd_view)

    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    args.cor = normalize_cor(args.lamb, args.cor)
    args.func(args)


if __name__ == "__main__":
    main()
