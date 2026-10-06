#!/usr/bin/env bash
# veloVI 0.3.1 (nsv_env environment): train, then compute velocities.
set -euo pipefail
cd "$(dirname "$0")"
${PY:-python} velovi_run_pipeline.py velovi_config.yaml
${PY:-python} compute_velovi_velocity_run.py compute_velovi_velocity_config_template.yaml
