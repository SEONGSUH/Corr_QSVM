## Code for Exploiting Feature Correlation for ZZFeatureMap-Based Quantum Support Vector Machines

## Directory structure

```
code/
├── README.md
├── LICENSE.txt       
├── requirements_A.txt               # Qiskit 1.4.6, PyTorch 2.9.1
├── requirements_B.txt               # Qiskit 2.3.1, PyTorch 2.10.0
├── save_ibm_account.py              # IBM Quantum credential setup
├── training/                       
│   ├── common.py                    
│   ├── train_cnn.py                 # 1. CNN feature extractor (+ MLP baseline)
│   ├── run_qsvm_noiseless.py        # 2. Noiseless-simulator QSVM/SVM
│   ├── run_qsvm_noisy.py            # 3. Noisy-simulator QSVM 
│   └── run_qsvm_qpu.py              # 4. QPU inference 
├── matlab_scripts/                  # MATLAB stages
│   ├── kta_simulation/              # 5. analytic KTA vs. Cor
│   │   ├── compute_KTA.m
│   │   └── visualization.m
│   └── results_analysis/            # 6. statistics and figures from the Excel outputs
│       ├── analysis.m
│       └── visualization.m
├── data/                            # torchvision download cache (auto-created)
└── results/                         # all outputs (auto-created)
   └── Fashion_59
      ├── complex_BatchNorm_seed0                      # CNN checkpoints only
      │   └── dim3 ...
      ├── complex_BatchNorm_seed0_Truncated            # noiseless QSVM
      │   └── dim3
      │       ├── qsvm
      │       │   ├── Z
      │       │   └── ZZ
      │       └── svm
      └── complex_BatchNorm_seed0_Truncated_NoisySim   # noisy QSVM
```

## Environments

Two Python environments were used (both Windows 11, Python 3.13). All scripts live in `training/`, but they do **not** share an environment. pick the one matching the script you are running:

| | Used by | Key packages |
|---|---|---|
| **A** (`requirements_A.txt`) | `train_cnn.py`, `run_qsvm_noiseless.py` | Qiskit 1.4.6, PyTorch 2.9.1 |
| **B** (`requirements_B.txt`) | `run_qsvm_noisy.py`, `run_qsvm_qpu.py` | Qiskit 2.3.1, PyTorch 2.10.0 |

Install, e.g.:

```
pip install -r requirements_A.txt  
pip install -r requirements_B.txt       # Separate venv!!
```

`run_qsvm_noiseless.py` relies on Qiskit 1.x primitives (`qiskit_algorithms`,
`qiskit_machine_learning`) that were removed or changed in Qiskit 2.x, so the two environments cannot be merged.

## Data

Dataset codes used throughout (suffix digits = class indices of the parent dataset), with the paper's labels:

| Code | Paper | Classes |
|---|---|---|
| `Fashion_59` | (a) | Fashion-MNIST sandal (5) vs. ankle boot (9) |
| `Fashion_24` | (b) | Fashion-MNIST pullover (2) vs. coat (4) |
| `CIFAR_02` | (c) | CIFAR-10 airplane (0) vs. bird (2) |
| `CIFAR_89` | (d) | CIFAR-10 ship (8) vs. truck (9) |
| `Number_35` | (e) | MNIST 3 vs. 5 |

## How to run

The grid is 5 datasets x d in {3..7} x lambda in {1, 0} x Cor in {0.0, 0.5,1.0} (when lambda = 1) x 5 folds, run for each of the two backbones and each of the three seeds. Outputs are written under `code/results/`.

Stages 1-3 sweep the full grid with no arguments, and accept `--arch`, `--seed`, `--dataset` to run only part of it. Stage 4 runs on hardware and is therefore restricted to one grid position at a time. 

Run the stages in order because each one reads what the previous wrote.

1. **CNN training** (environment A): trains the feature extractor with `L = L_cls + lambda * L_corr` and writes checkpoints plus `summary.xlsx`, whose `test_acc` column is the **MLP baseline** reported in the paper.
```
python training/train_cnn.py
python training/train_cnn.py --arch complex --seed 0
```
2. **Noiseless QSVM/SVM** (environment A): loads the checkpoints, extracts features, scales them to [0, pi], and evaluates the SVM-RBF baseline and the Z/ ZZ feature-map QSVMs with statevector kernel fidelities.
```
python training/run_qsvm_noiseless.py
python training/run_qsvm_noiseless.py --arch complex --seed 0
```
3. **Noisy QSVM** (environment B): same pipeline, but kernel entries are estimated from 1024 shots per overlap circuit on `AerSimulator.from_backend(ibm_fez)`. This stage needs IBM Quantum credentials, used **only** to download the ibm_fez noise model, and nothing is executed on hardware (no token consumed).
Run the credential setup once (in environment B). It prompts for your API token with hidden input and for the instance CRN, stores them in the  qiskit-ibm-runtime account file (`~/.qiskit/qiskit-ibm.json`), and verifies that ibm_fez is reachable:
```
python save_ibm_account.py
```
Afterwards `QiskitRuntimeService()` picks the account up with no arguments, and the setup does not need to be repeated:
```
python training/run_qsvm_noisy.py
python training/run_qsvm_noisy.py --arch complex --seed 0
```
4. **QPU inference** (environment B): only inferences the kernel on real quantum hardware based on kernels trained on noiseless simulator. One grid position per run, picked with `--dataset/--arch/--seed/--dim/--lamb/--cor/--fold/--fm` (defaults: `--dim 3 --lamb 1 --cor 0.0 --fold 0 --fm ZZ`). 

`--mode statevector` measures the same kernel locally, without credentials or QPU time. Run it first as a rehearsal and check that it prints `[check] nearest-neighbour distance = ... -> OK`:
```
python training/run_qsvm_qpu.py run --dataset Fashion_59 --arch lightweight --seed 0 --mode statevector --rows 0
```
`--mode qpu` submits to hardware, `--rows` test samples per call out of 50. Each job is saved to `<TAG>_Kqpu.npz` as it returns, so re-running continues from the rows still missing. `view` shows progress and scores the finished rows:
```
python training/run_qsvm_qpu.py run --dataset Fashion_59 --arch lightweight --seed 0 --mode qpu --rows 2
python training/run_qsvm_qpu.py view --dataset Fashion_59 --arch lightweight --seed 0
```

## MATLAB analysis

`matlab_scripts/` needs MATLAB (developed upon R2024B version) environment .

5. **KTA simulation** (`kta_simulation/`): analytic KTA of the ZZ feature map as a function of Cor, with no experiment data involved.
- `compute_KTA.m` sweeps dim x angle x seed x Cor and writes `random_vectors_angle_<angle>/kernel_statistics_<d>_seed<n>.mat`. Combinations already on disk are skipped, so an interrupted sweep can just be re-run.
- `visualization.m` averages those over seeds, combines the moments into the analytic KTA, and saves one `KTA_Cor_Angle_<angle>.jpg` per angle. 

6. **Results analysis** (`results_analysis/`): statistics and figures from the Excel files the Python stages wrote. Set `cfg.RESULT_ROOT` to the `results/` tree.
- `analysis.m` runs one of three analyses, selected by `analyses`: `'mac_acc'` (Accuracy ~ MAC regression, per-Cor accuracy, Wilcoxon test of Cor = 0.0 vs. 1.0), `'mlp'` (the same per-Cor statistics for the MLP baseline), `'mac_kta_acc'` (empirical KTA from the stored kernel statistics, then KTA ~ MAC and Accuracy ~ KTA). Choose the classifier with `cfg.target_classifier_name` (`'ZZ' | 'Z' | 'rbf'`) and the noiseless or noisy results with `cfg.norm_tail` (`'Truncated' | 'Truncated_NoisySim'`).
- `visualization.m` draws the grouped accuracy bar charts of the paper. Its numbers are transcribed from `analysis.m`'s reports.

## License

This code is released under the MIT License (see `LICENSE`).
