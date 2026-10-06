# Third-party licenses

## Code included in this folder (copied or adapted)

| Project | License | File | Where it is used |
|---|---|---|---|
| scvi-tools (https://github.com/scverse/scvi-tools) | BSD 3-Clause | `scvi_tools_LICENSE` | Model and module classes of both VAEs and the helper distributions in `nsv/` |
| scVelo (https://github.com/theislab/scvelo) | BSD 3-Clause | `scvelo_LICENSE` | `nsv_runs/scv_velocity_graph_new.py` (modified `velocity_graph`) and `velocity_metrics/velocity_confidence_scaled.py` (modified `velocity_confidence`) |
| VeloAE (https://github.com/qiaochen/VeloAE) | MIT | `veloae_LICENSE` | Cross-boundary direction and in-cluster coherence helpers in `velocity_metrics/compute_cbdir_run.py` (`keep_type`, `remove_type`, the metric definitions) |

## Packages run by the scripts (dependencies, not redistributed)

| Package | License | File | Used by |
|---|---|---|---|
| scanpy | BSD 3-Clause | `scanpy_LICENSE` | all steps |
| anndata | BSD 3-Clause | `anndata_LICENSE` | all steps |
| veloVI (https://github.com/YosefLab/velovi) | BSD 3-Clause | `velovi_LICENSE` | `other_methods/velovi_runs/` |
| VeloVAE (https://github.com/welch-lab/VeloVAE) | BSD 3-Clause | `velovae_LICENSE` | `other_methods/velovae_runs/` |
| UniTVelo (https://github.com/StatBiomed/UniTVelo) | BSD 3-Clause | `unitvelo_LICENSE` | `other_methods/unitvelo_runs/` |
| cellDancer (https://github.com/GuangyuWangLab2021/cellDancer) | BSD 3-Clause | `celldancer_LICENSE` | `other_methods/celldancer_runs/` |

TFvelo (https://github.com/xiaoyeye/TFvelo) was run from its repository, which
has no license file; none of its code is included here. The license texts are
copied unchanged from the package distributions (scvi-tools 1.0.4, scVelo
0.3.4, scanpy 1.9.3, anndata 0.8.0) or the projects' GitHub repositories.
