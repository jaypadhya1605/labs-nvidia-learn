#!/usr/bin/env bash
# Run ON the CVM after installing the driver with Microsoft's confidential-GPU onboarding package
# (github.com/Azure/az-cgpu-onboarding - follow its README for your image; it enables H100 CC mode).
set -euo pipefail
nvidia-smi conf-compute -f        # expect: CC status: ON
nvidia-smi conf-compute -grs      # expect: ready
python3 -m venv ~/att && . ~/att/bin/activate
pip install -q nv-local-gpu-verifier
python3 -m verifier.cc_admin --user_mode   # GPU evidence + certificate chain + RIM measurements -> attestation result
