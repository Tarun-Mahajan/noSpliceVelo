#!/usr/bin/env bash
# Step 2: velocities, fitted moments and state reductions from the trained models.
set -euo pipefail
cd "$(dirname "$0")"
python compute_velo_run.py "${1:-compute_velo_config_template_.yaml}"
