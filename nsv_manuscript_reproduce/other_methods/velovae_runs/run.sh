#!/usr/bin/env bash
# VeloVAE 0.1.2 (nsv_env environment).
set -euo pipefail
cd "$(dirname "$0")"
${PY:-python} run_velovae_batch.py --config velovae_config_example.yaml
