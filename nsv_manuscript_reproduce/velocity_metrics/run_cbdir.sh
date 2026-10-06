#!/usr/bin/env bash
# Step 5: CBDir and ICCoh for all methods and datasets (run from velocity_metrics/).
set -euo pipefail
cd "$(dirname "$0")"
python compute_cbdir_run.py "${1:-cbdir_global_config_4throot.yaml}"
