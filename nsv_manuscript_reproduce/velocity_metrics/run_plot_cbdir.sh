#!/usr/bin/env bash
# Step 6: CBDir/ICCoh figures and statistics (run from velocity_metrics/).
set -euo pipefail
cd "$(dirname "$0")"
python plot_cbdir.py --global-config "${1:-cbdir_global_config_4throot.yaml}" \
    --plot-config "${2:-plot_cbdir_config_nsv_comparison.yaml}"
