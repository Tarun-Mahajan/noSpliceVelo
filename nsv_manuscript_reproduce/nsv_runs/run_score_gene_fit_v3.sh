#!/usr/bin/env bash
# Step 3: gene-level goodness of fit (R2 vs mean null, quadratic-vs-line, bow, signed separation).
set -euo pipefail
cd "$(dirname "$0")"
python score_gene_fit_v3_run.py "${1:-score_gene_fit_v3_config_template_.yaml}"
