"""
One-time IBM Quantum credential setup for training/run_qsvm_noisy.py.

    python save_ibm_account.py

Prompts for the API token (hidden input) and the instance CRN, then stores them
in the qiskit-ibm-runtime account file (~/.qiskit/qiskit-ibm.json). Nothing is
written into this repository, so the token cannot leak with the supplement.

Afterwards `QiskitRuntimeService()` in run_qsvm_noisy.py finds the account with
no arguments. Run this in environment B (requirements_B.txt).
"""

import getpass

from qiskit_ibm_runtime import QiskitRuntimeService

CHANNEL = "ibm_cloud"   

token = getpass.getpass("IBM Quantum API token (input hidden): ").strip()
if not token:
    raise SystemExit("No token entered; nothing saved.")

instance = input("Instance CRN (press Enter to skip): ").strip()

kwargs = dict(channel=CHANNEL, token=token, overwrite=True)
if instance:
    kwargs["instance"] = instance

QiskitRuntimeService.save_account(**kwargs)
print("Saved accounts:", list(QiskitRuntimeService.saved_accounts()))

# Verify the credentials actually work and the target backend is reachable
service = QiskitRuntimeService()
backend = service.backend("ibm_fez")
print(f"OK - connected to {backend.name} ({backend.num_qubits} qubits)")
