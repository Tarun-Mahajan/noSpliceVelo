#!/usr/bin/env bash
# Step 4: velocity-gene filter, velocity graph, streams and pseudotime.
#   bash run_generate_velo_stream.sh          # noSpliceVelo only
#   bash run_generate_velo_stream.sh all      # noSpliceVelo + the seven compared methods
set -euo pipefail
cd "$(dirname "$0")"
python generate_velo_stream_run.py velo_stream_config_template_nsv.yaml
if [ "${1:-}" = "all" ]; then
  for m in scvelo scvelo_stoch velovi velovae celldancer unitvelo tfvelo; do
    python generate_velo_stream_run.py "velo_stream_config_template_${m}_4throot.yaml"
  done
fi
