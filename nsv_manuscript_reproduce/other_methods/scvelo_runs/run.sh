#!/usr/bin/env bash
# scVelo 0.3.4, dynamical and stochastic modes (nsv_env environment).
set -euo pipefail
cd "$(dirname "$0")"
${PY:-python} scvelo_run_pipeline.py scvelo_config.yaml
