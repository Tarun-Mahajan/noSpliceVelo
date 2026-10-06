#!/usr/bin/env bash
# cellDancer 1.1.7 in its own Python 3.7 environment (torch 1.10.0, scvelo 0.2.5):
#   PY=/path/to/celldancer_env/bin/python bash run.sh
#   PY=... bash run.sh --only human_bonemarrow     # one dataset
#   PY=... bash run.sh --rebuild                   # rebuild the h5ad from an existing cellDancer CSV
set -euo pipefail
cd "$(dirname "$0")"
${PY:-python} run_celldancer_batch.py --config celldancer_config.yaml "$@"
