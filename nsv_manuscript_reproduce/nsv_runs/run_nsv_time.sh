#!/usr/bin/env bash
# Step 1: train SCVIModified + model selection + noSpliceVelo for every dataset in the config.
# Activate the environment first (environment_gpu_final.yml). Run from nsv_runs/.
set -euo pipefail
cd "$(dirname "$0")"
python nsv_run_pipeline.py "${1:-nsv_config_runs_studentT_time_template.yaml}"
