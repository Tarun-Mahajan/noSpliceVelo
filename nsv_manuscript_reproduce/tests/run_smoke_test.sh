#!/usr/bin/env bash
# End-to-end smoke test of the noSpliceVelo manuscript pipeline on synthetic data.
#
#   bash tests/run_smoke_test.sh            # from nsv_manuscript_reproduce/
#   START=4 bash tests/run_smoke_test.sh    # resume at step 4 (models already trained)
#   PY=/path/to/python bash tests/run_smoke_test.sh
#
# Runs every stage with the manuscript settings, except that training epochs are
# capped (see tests/make_test_configs.py). CPU-only is fine: about 45 min on
# 2 cores, most of it in step 1; a GPU is used automatically if present.
# Outputs go to data/dummy_* and tests/results_*; logs to tests/logs/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python}
START=${START:-0}
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$ROOT/tests/logs"
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
STREAM=nosplicevelo_score_gene_fit_v3_4throot_type1_vote

cd "$ROOT/tests"
if [ "$START" -le 0 ]; then
  step "0. synthetic inputs"
  $PY make_dummy_adata.py --out ../data/dummy_a/adata_pan.h5ad --seed 0
  $PY make_dummy_adata.py --out ../data/dummy_b/adata_pan.h5ad --seed 1
fi
step "configs (from the manuscript templates)"
$PY make_test_configs.py

cd "$ROOT/nsv_runs"
if [ "$START" -le 1 ]; then
  step "1. train SCVIModified + model selection + noSpliceVelo"
  $PY nsv_run_pipeline.py ../tests/configs/nsv_dummy.yaml
fi
if [ "$START" -le 2 ]; then
  step "2. velocities, fitted moments, state reductions"
  $PY compute_velo_run.py ../tests/configs/compute_velo_dummy.yaml
fi
if [ "$START" -le 3 ]; then
  step "3. gene-level goodness of fit"
  $PY score_gene_fit_v3_run.py ../tests/configs/score_gene_fit_v3_dummy.yaml
fi
if [ "$START" -le 4 ]; then
  step "4. gene filter, velocity graph, streams, pseudotime"
  $PY generate_velo_stream_run.py ../tests/configs/velo_stream_dummy.yaml
fi
if [ "$START" -le 5 ]; then
  step "5. CBDir / ICCoh"
  cd "$ROOT/tests"
  $PY make_shuffled_control.py ../data/dummy_a_studentT_time/$STREAM ../data/dummy_b_studentT_time/$STREAM
  cd "$ROOT/velocity_metrics"
  $PY compute_cbdir_run.py ../tests/configs/cbdir_global_dummy.yaml
fi
if [ "$START" -le 6 ]; then
  step "6. CBDir figures, statistics and Table S2"
  cd "$ROOT/velocity_metrics"
  $PY plot_cbdir.py --global-config ../tests/configs/cbdir_global_dummy.yaml \
                    --plot-config ../tests/configs/plot_cbdir_dummy.yaml
  $PY benchmark_table_s2.py --global-config ../tests/configs/cbdir_global_dummy.yaml \
                            --out-dir ../tests/results_table_s2
fi
if [ "$START" -le 7 ]; then
  step "7. gene-level direction and VAE vs naive moments"
  cd "$ROOT/analyses/gene_direction"
  $PY gene_direction_benchmark.py ../../tests/configs/gene_direction_dummy.yaml
  cd "$ROOT/analyses/mu_var_corr"
  $PY mu_var_corr.py ../../tests/configs/mu_var_corr_dummy.yaml
fi

step "8. check outputs"
cd "$ROOT/tests"
$PY check_smoke_test.py
