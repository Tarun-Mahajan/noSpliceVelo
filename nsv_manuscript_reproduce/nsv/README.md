# `nsv/`: model code

This is the library code that the drivers in `../nsv_runs/` import. No file
here is run directly. The drivers put this folder on `sys.path` themselves
(override with the `NSV_SRC` environment variable). Both models are
`scvi-tools` 1.0.4 models (`BaseModelClass` + `VAEMixin`), so they are trained,
saved and loaded with the usual `scvi-tools` API.

| File | Role | Methods section |
|---|---|---|
| `scvi_modified_capture_efficiency_model.py` | `SCVIModified`: the first VAE (model class: `setup_anndata`, training, `get_likelihood_parameters_new` returning per-cell `mu`, `var`, `library`) | Estimating mean and variance of expression via mRNA count modeling |
| `scvi_modified_capture_efficiency_module.py` | Generative and inference networks of the first VAE: per-cell capture efficiency, per-cell/gene mean and variance, the correlation loss toward naive kNN moments | Parameterization / Loss function of the first VAE |
| `ellipse_fit.py` | `VelocityModelSelector`: per gene, fits a line, a parabola and an ellipse to the smoothed (mean, variance) cloud, picks one by BIC (margins `thresh_p=2`, `thresh_e=5` as called by the pipeline), labels noise genes (adj. R² < 0.1 or slope < 1), and writes `var['best_fit']` and the empirical branch assignment `layers['branch_assignment']` | Data preprocessing, step 5 |
| `nosplicevelo_model_v5_polar.py` | `noSpliceVelo`: the second VAE (model class) | noSpliceVelo model specification / inference procedure |
| `nosplicevelo_module_v5_polar.py` | `VAENoiseVelo`: generative process (4 kinetic states: up, upper steady state, down, lower steady state), polar features of the variance-vs-mean plane, Student-t likelihood with df annealed from 1000 to 4, Dirichlet state prior, empirical corners and branch prior, loss terms (λ1–λ3, λ_sep, λ_ω) | Generative process, polar representation, loss function |
| `nosplicevelo_ll_params.py` | Batched posterior-predictive inference used by `compute_velo_run.py`: 10 samples, per-sample argmax state, fitted mean/variance, latent time, velocity of mean and variance, and the state reductions | Downstream tasks |
| `state_reductions.py` | Alternative ways to combine the 4 per-state velocities (`argmax_knn`, `tempered`) and the kNN posterior smoothing used by `vote_knn`; diagnostics only, the manuscript uses `vote` | Downstream tasks |
| `scvi_clonealign_distributions.py`, `scvi_clonealign_layers.py`, `scvi_clonealign_losses.py` | Helper distributions/layers/losses imported by the two modules | – |
| `constants_tmp.py` | Registry keys for the extra AnnData fields (mean/std layers, priors) | – |

## The time-dependence factor ω

In `VAENoiseVelo.generative` (search for `use_time_dependence`):

```python
if self.use_time_dependence:
    b_t = F.sigmoid(self.b_t(px)) + 1e-4
    f_t = F.sigmoid(self.b_t(px)) + 1e-4      # same network: f and B share one factor
    # f_t = F.sigmoid(self.f_t(px)) + 1e-4    # alternative: separate factor for f (not used)
else:
    b_t = torch.ones_like(px_B2_gene)
    f_t = torch.ones_like(px_B2_gene)
px_B2_gene = px_B2_gene * b_t                 # burst size of the final up-branch state
px_f2_by_gam_gene = px_f2_by_gam_gene * f_t   # burst frequency of the final up-branch state
```

The penalty `time_loss * (-log b_t - log f_t)` keeps ω near 1. The model
default is `use_time_dependence=False`. The pipeline passes the config value,
whose default is `True`, as in the manuscript.

## Velocity output names

`compute_velo_run.py` writes these layers (per cell and gene) to `adata_nosplicevelo_vote.h5ad`:

| Layer | Meaning |
|---|---|
| `mu_scvi_smooth`, `var_scvi_smooth` | Smoothed mean and variance from the first VAE: the data the second VAE fits |
| `mu_naive_smooth`, `var_naive_smooth` | Naive kNN moments of the raw counts (k = 30 on 50 PCs, two rounds of averaging) |
| `mu_fit`, `var_fit` | Fitted mean and variance of the most probable state |
| `velocity_mu`, `velocity_var` | Velocity of mean/variance of the most probable state (`argmax_stable`: state probabilities averaged over the 10 samples, then argmax) |
| `velocity_mu_vote`, `velocity_var_vote` | **Used in the manuscript**: argmax state in each of the 10 samples, its velocity, averaged over samples (Methods, Downstream tasks, steps 1–4) |
| `velocity_mu_soft`, `*_tempered2`, `*_vote_knn` | Other reductions, kept for diagnostics (the capture-efficiency experiment uses `velocity_mu_soft`) |
| `time_latent` | Latent time per cell and gene (used by the MURK filter) |

Per gene (`var`): `best_fit` (line / parabola / ellipse), `time_switch`,
`gamma_mRNA`, and the steady-state moments `mu_0_gene`, `mu_up_f_gene`, and so on.
