#!/usr/bin/env bash
# UniTVelo 0.2.5.2. Set PY to the python of the environment that has unitvelo.
set -euo pipefail
cd "$(dirname "$0")"
${PY:-python} run_untivelo_batch.py --config unitvelo_config.yaml
