#!/usr/bin/env bash
# Capture-bias robustness runs for pancreas on 3 GPUs.
#
#   GPU 0: reviewer_s0      -> reviewer_s1
#   GPU 1: uniform_s0       -> uniform_s1
#   GPU 2: gradient_down_r4 -> gradient_down_r4_s1
#
# Wave 1 (the first dataset on each GPU) is reviewer_s0 | uniform_s0 | gradient_down_r4,
# i.e. a complete pseudotime-dependent / control / stress-test comparison on its own; wave 2 adds
# the second thinning seed of each. Existing inputs are not rewritten (--skip_existing).
# The manuscript used reviewer_s0/s1 and uniform_s0/s1 (Supplementary Fig. on capture
# efficiency); gradient_down_r4(_s1) is an additional stress test.
# Everything runs from nsv_runs/, so all relative paths below are relative to nsv_runs/.
# Override any setting with an environment variable, e.g.
#   PY=/path/to/env/bin/python OUT=../data/Pancreas_capture_bias bash run_capture_bias.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../nsv_runs"

PY=${PY:-python}
PIPE=${PIPE:-nsv_run_pipeline.py}
OUT=${OUT:-../data/Pancreas_capture_bias}
ADATA=${ADATA:-../data/Pancreas/adata_pan.h5ad}
# nSV stream output of the full pancreas data: supplies the pseudotime used by the
# pseudotime-dependent rule (reviewer_*), the original embedding and the original velocity-gene set
REF=${REF:-../data/Pancreas_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote/adata_nosplicevelo_stream.h5ad}
SESSION=${SESSION:-capbias}
T="OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMBA_NUM_THREADS=14 LOKY_MAX_CPU_COUNT=14 PYTHONUNBUFFERED=1 MPLBACKEND=Agg"
mkdir -p logs

# 1) thinned inputs for all six conditions (CPU only, a few minutes)
"$PY" "$HERE/make_capture_bias_datasets.py" \
    --adata "$ADATA" --reference "$REF" --out_root "$OUT" --skip_existing \
    --conditions reviewer_s0 uniform_s0 gradient_down_r4 reviewer_s1 uniform_s1 gradient_down_r4_s1

# 2) one nSV config per GPU, wave-1 condition first
"$PY" - "$OUT" <<'EOF'
import os, sys, yaml
out = sys.argv[1]
cfg = yaml.safe_load(open(os.path.join(out, "nsv_config_capture_bias.yaml")))
by = {d["name"].replace("pancreas_", "", 1): d for d in cfg["datasets"]}
groups = {0: ["reviewer_s0", "reviewer_s1"], 1: ["uniform_s0", "uniform_s1"], 2: ["gradient_down_r4", "gradient_down_r4_s1"]}
for g, conds in groups.items():
    c = dict(cfg)
    c["log_file"] = f"./logs/nsv_pipeline_capture_bias_gpu{g}.log"
    c["datasets"] = [by[k] for k in conds]
    p = os.path.join(out, f"nsv_config_capture_bias_gpu{g}.yaml")
    yaml.safe_dump(c, open(p, "w"), sort_keys=False)
    print(f"GPU {g}: {conds} -> {p}")
EOF

# 3) launch, one tmux window per GPU
tmux new-session -d -s "$SESSION" -n gpu0 -c "$PWD"
for g in 0 1 2; do
  [ "$g" -gt 0 ] && tmux new-window -t "$SESSION" -n "gpu$g" -c "$PWD"
  tmux send-keys -t "$SESSION:gpu$g" \
    "CUDA_VISIBLE_DEVICES=$g $T $PY $PIPE $OUT/nsv_config_capture_bias_gpu$g.yaml 2>&1 | tee logs/capbias_gpu$g.out" Enter
done
echo "launched in tmux session '$SESSION' (tmux attach -t $SESSION)"
echo "next, once all six finish (from nsv_runs/):"
echo "  python compute_velo_run.py      $OUT/compute_velo_config_capture_bias.yaml"
echo "  python score_gene_fit_v3_run.py $OUT/score_gene_fit_v3_config_capture_bias.yaml"
echo "  python generate_velo_stream_run.py $OUT/velo_stream_config_capture_bias.yaml"
echo "  python ../velocity_metrics/compute_cbdir_run.py $OUT/cbdir_capture_bias/cbdir_global_capture_bias.yaml"
