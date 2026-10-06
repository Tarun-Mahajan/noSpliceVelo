"""
nosplicevelo_ll_params.py
=========================

Standalone, GPU/memory-optimized extraction of `get_nosplicevelo_ll_params`
and everything it depends on, pulled out of `utils_RNA_velocity.py`.

Why this rewrite exists
-----------------------
The original `get_nosplicevelo_ll_params_geneCell` (and `_gene`) call the
model's likelihood-parameter methods on ALL cells at once:

    params_ = model_.get_likelihood_parameters_new()          # (N_cells, N_genes[, 4])

For 18k cells x 2k genes with 4 latent states, a *single* prob_state array is
18000 * 2000 * 4 * 4 bytes = ~1.1 GB, and the forward pass allocates several
such tensors on the GPU simultaneously -> GPU OOM -> the kernel dies on HPC.
On top of that, the CPU-side `np.stack(..., axis=2)` builds five more
(N, G, 4) arrays (~5-6 GB of host RAM).

The fix is to run inference over cells in BATCHES and never hold the full
(N, G, 4) tensors in memory. This module does that while keeping the numerical
result identical to the original (same argmax-of-prob_state selection, same
bootstrap averaging).

--------------------------------------------------------------------------
MODEL REQUIREMENT (read this)
--------------------------------------------------------------------------
Chunking cells only bounds GPU memory if the model can run inference on a
*subset* of cells. There are three tiers, best first:

  (A) `get_likelihood_parameters_new` already accepts `batch_size=` and loops
      internally with a DataLoader (standard scvi-tools pattern). Then the
      simplest possible fix is to just pass a batch_size -- no external
      chunking needed. This module auto-detects and uses it.

  (B) The method accepts `indices=` (and optionally `adata=`) so we can ask it
      for one block of cells at a time. This module chunks over `indices`.

  (C) The method takes no such args and always processes `model.adata` whole.
      Then you MUST either (i) add batch support to the method (a ~10 line
      patch, see `MODEL_PATCH_EXAMPLE` at the bottom of this file), or
      (ii) let this module fall back to subsetting the AnnData per chunk via
      `adata=`. Fallback (ii) works only if the method accepts `adata=`.

Use `describe_model_support(model_)` to print which tier your model is in.

Author: extracted/optimized for Tarun's HPC run.
"""

from __future__ import annotations

import gc
import inspect
from typing import Optional, Sequence

import numpy as np
import torch

try:
    import scvelo as scv
except Exception:  # pragma: no cover - scvelo optional at import time
    scv = None

from scipy.sparse import issparse


# =============================================================================
# Configuration
# =============================================================================

# Outer cell-chunk size. Each chunk is one call into the model's likelihood
# method (via `indices`), so the model returns / concatenates only this many
# cells at a time -> bounds HOST RAM. ~4096 keeps peak host memory small while
# amortizing Python overhead.
DEFAULT_CELL_BATCH = 4096

# Inner GPU minibatch size passed straight to the model's dataloader
# (`batch_size=`). The model's own `_make_data_loader` loop uses this for the
# forward pass, so this is what bounds GPU memory. Lower it if you still OOM on
# the device; raise it on A100/H100. `None` -> scvi.settings.batch_size (~128).
DEFAULT_GPU_BATCH = 512

# Accumulate bootstrap sums in this dtype. float32 halves host RAM vs float64
# and is plenty for downstream velocity analysis. Set to np.float64 if you
# need bit-for-bit parity with the original.
ACCUM_DTYPE = np.float32

# State order used by the original code when stacking. DO NOT reorder:
# index 0 = "up", 1 = "up_f" (steady up), 2 = "down", 3 = "down_f" (steady down)
_N_STATES = 4


# =============================================================================
# Small numerical helpers (unchanged behavior)
# =============================================================================

def get_mu_var_time(mu_0, var_0, mu_f, var_f, gamma_, time):
    """Analytic mean/variance of the burst model at a given time.

    Identical to the original in utils_RNA_velocity.py. Works on numpy arrays.
    """
    p_t = np.exp(-gamma_ * time)
    mu_t = mu_0 * p_t + mu_f * (1 - p_t)
    var_t = (var_0 - mu_0) * p_t ** 2.0 + (var_f - mu_f) * (1 - p_t ** 2.0) + mu_t
    return mu_t, var_t


def _select_by_state(max_idx: np.ndarray, state_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Pick, per (cell, gene), the value of the argmax state.

    Equivalent to the original:
        stacked = np.stack(state_arrays, axis=2)              # (N, G, 4)
        out = stacked[arange(N)[:,None], arange(G)[None,:], max_idx]
    but WITHOUT materializing the (N, G, 4) array. Uses boolean-mask writes so
    peak extra memory is one (N, G) array instead of four.

    `state_arrays` must be in canonical order [up, up_f, down, down_f].
    """
    out = np.zeros_like(state_arrays[0])
    for k, arr in enumerate(state_arrays):
        m = (max_idx == k)
        # arr may be a constant broadcast (e.g. tmax*ones); handle scalars too
        out[m] = arr[m] if np.ndim(arr) else arr
    return out


def _mixture_by_state(probs: np.ndarray, state_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Posterior-weighted average over states: sum_s p_s * arr_s.

    The 2nd VAE reconstructs the (mu, var) TARGETS through the state posterior;
    it does NOT model counts, so each reconstructed quantity is just the
    probability-weighted mixture average over states (no law-of-total-variance /
    between-state term). This is the posterior-mean reconstruction, which is
    unbiased for the target, unlike the argmax (mode) used by _select_by_state.

    probs        : (n, G, S) per-state posterior probabilities.
    state_arrays : sequence of S arrays, each (n, G) (constants allowed),
                   in canonical order [up, up_f, down, down_f].
    Memory: one (n, G) accumulator; no (n, G, S) stack.
    """
    out = np.zeros_like(state_arrays[0], dtype=state_arrays[0].dtype)
    for k, arr in enumerate(state_arrays):
        out += probs[:, :, k] * arr
    return out


def compute_gene_velocity_consistency(adata, velo_, n_neighbors=30):
    """Per-gene neighborhood coherence of a velocity matrix.

    Unchanged from the original. Must be run ONCE on the full (N, G) velocity
    matrix (it uses the full kNN graph), so it lives outside the cell-chunk
    loop.
    """
    if 'neighbors' not in adata.uns:
        if scv is None:
            raise RuntimeError("scvelo is required to build the neighbor graph")
        scv.pp.neighbors(adata, n_neighbors=n_neighbors)

    V = velo_.copy()
    if issparse(V):
        V = V.toarray()

    W = adata.obsp['connectivities']
    W_norm = W.multiply(1 / W.sum(axis=1))
    V_smooth = W_norm @ V
    V_smooth = np.asarray(V_smooth)

    gene_scores = []
    for i in range(V.shape[1]):
        corr = np.corrcoef(V[:, i], V_smooth[:, i])[0, 1]
        gene_scores.append(corr)
    return np.nan_to_num(gene_scores)


# =============================================================================
# Model introspection + chunked inference wrapper
# =============================================================================

def _method_params(model_, method_name: str):
    fn = getattr(model_, method_name, None)
    if fn is None:
        raise AttributeError(f"model has no method {method_name!r}")
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def describe_model_support(model_, method_name: str = "get_likelihood_parameters_new") -> dict:
    """Report how the model's likelihood method can be chunked. Prints a hint."""
    params = _method_params(model_, method_name)
    support = {
        "has_batch_size": "batch_size" in params,
        "has_indices": "indices" in params,
        "has_adata": "adata" in params,
        "params": sorted(params),
    }
    if support["has_indices"] and support["has_batch_size"]:
        tier = ("A+ (indices + batch_size) -- BEST: outer chunk over indices "
                "bounds host RAM, inner batch_size bounds GPU RAM")
    elif support["has_indices"]:
        tier = "A (indices) -- outer chunk over cell indices bounds host RAM"
    elif support["has_batch_size"]:
        tier = ("B (batch_size only) -- GPU bounded by dataloader, but the full "
                "result is assembled in host RAM in one call")
    elif support["has_adata"]:
        tier = "C-fallback (adata) -- chunk by subsetting AnnData"
    else:
        tier = "D (no chunk args) -- PATCH THE MODEL METHOD (see MODEL_PATCH_EXAMPLE)"
    support["tier"] = tier
    print(f"[{method_name}] chunking tier: {tier}")
    print(f"[{method_name}] signature params: {support['params']}")
    return support


def _infer_ll_params(
    model_,
    method_name: str,
    *,
    indices: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
    adata=None,
):
    """Call a model likelihood method passing only the kwargs it supports.

    Wrapped in torch.no_grad() to avoid building the autograd graph (saves a
    large chunk of GPU memory during inference).
    """
    fn = getattr(model_, method_name)
    params = _method_params(model_, method_name)
    kwargs = {}
    if indices is not None and "indices" in params:
        kwargs["indices"] = indices
    if batch_size is not None and "batch_size" in params:
        kwargs["batch_size"] = batch_size
    if adata is not None and "adata" in params:
        kwargs["adata"] = adata
    with torch.no_grad():
        return fn(**kwargs)


def _iter_cell_chunks(
    model_,
    method_name: str,
    n_cells: int,
    cell_batch_size: int,
    gpu_batch_size: Optional[int] = None,
    adata=None,
):
    """Yield (chunk_indices, params_dict) covering all cells exactly once.

    Strategy chosen from model capabilities (best first):
      * indices [+ batch_size]: outer chunk over `indices` (bounds HOST RAM);
        `batch_size` is forwarded so the model's own dataloader minibatches the
        GPU forward within each chunk (bounds GPU RAM). This is the path this
        model (nosplicevelo_model_v4_polar) takes.
      * batch_size only: one call; GPU is bounded by the dataloader but the full
        result is concatenated in host RAM.
      * adata only: outer chunk by subsetting the AnnData.
      * none: single whole-dataset call (may OOM) -> warn.
    """
    params = _method_params(model_, method_name)
    starts = range(0, n_cells, cell_batch_size)

    # Best: outer chunk over indices; inner GPU minibatch via batch_size.
    if "indices" in params:
        for s in starts:
            idx = np.arange(s, min(s + cell_batch_size, n_cells))
            yield idx, _infer_ll_params(
                model_, method_name, indices=idx,
                batch_size=gpu_batch_size, adata=adata,
            )
        return

    # GPU bounded internally, but host holds the whole result in one call.
    if "batch_size" in params:
        p = _infer_ll_params(model_, method_name, batch_size=gpu_batch_size, adata=adata)
        yield np.arange(n_cells), p
        return

    # Fallback: chunk by subsetting the AnnData.
    if "adata" in params and adata is not None:
        for s in starts:
            idx = np.arange(s, min(s + cell_batch_size, n_cells))
            sub = adata[idx].copy()
            yield idx, _infer_ll_params(model_, method_name, adata=sub)
            del sub
        return

    # No way to chunk -> single whole-dataset call (may OOM). Warn loudly.
    print(
        f"[WARN] {method_name} exposes no batch_size/indices/adata argument; "
        f"running on all {n_cells} cells at once. This is the configuration "
        f"that crashes the kernel. See MODEL_PATCH_EXAMPLE to fix it."
    )
    yield np.arange(n_cells), _infer_ll_params(model_, method_name)


def _to_np(x, dtype=ACCUM_DTYPE):
    """Copy a model output to a numpy array of the target dtype (off-GPU)."""
    if torch.is_tensor(x):
        x = x.detach().to("cpu").numpy()
    return np.asarray(x, dtype=dtype)


# =============================================================================
# gene-cell specific parameters (the function that was crashing)
# =============================================================================

def get_nosplicevelo_ll_params_geneCell(
    model_,
    model_0,
    adata_,
    nboot=10,
    prob_thresh=0.5,
    tmax=24.0,
    basis="umap",
    cell_batch_size: int = DEFAULT_CELL_BATCH,
    gpu_batch_size: Optional[int] = DEFAULT_GPU_BATCH,
    verbose: bool = True,
    state_reduction: str = "argmax_stable",
):
    """Gene-cell-specific velocity params, computed in cell batches.

    Note: the outputs of this function ('mu_all', 'var_all', 'velo_mu_all', ...)
    are NOT the saved fitted layers (those come from
    get_nosplicevelo_ll_params_gene); they feed velocity-consistency and
    diagnostics. `state_reduction` here selects "soft" (posterior mixture) vs a
    per-bootstrap argmax for those secondary outputs; the on-manifold
    "argmax_stable" reconstruction is applied where it matters, in
    get_nosplicevelo_ll_params_gene.

    Numerically equivalent to the original, but the per-boot work is done one
    cell-chunk at a time so the GPU never sees all 18k cells at once and the
    host never holds a full (N, G, 4) stack.

    Extra args:
        cell_batch_size : cells per outer chunk / model call (host RAM bound).
        gpu_batch_size  : inner dataloader batch_size for the GPU forward.
    """
    N = adata_.n_obs

    # Bootstrap accumulators (allocated lazily once we know G).
    acc = None

    for b_ in range(nboot):
        if verbose:
            print(f"gene-cell boot = {b_}")

        # Per-boot full-N buffers, filled chunk by chunk. Allocated on first
        # chunk when G is known.
        buf = None
        mu_moment_present = None

        for idx, params_ in _iter_cell_chunks(
            model_, "get_likelihood_parameters_new", N, cell_batch_size,
            gpu_batch_size=gpu_batch_size, adata=adata_,
        ):
            # ---- pull per-chunk arrays off the GPU immediately ----
            has_moment = 'mu' in params_
            mu_moment = _to_np(params_['mu']) if has_moment else None
            var_moment = _to_np(params_['var']) if has_moment else None

            velo_mu_up = _to_np(params_['velo_mu_up'])
            velo_var_up = _to_np(params_['velo_var_up'])
            velo_mu_down = _to_np(params_['velo_mu_down'])
            velo_var_down = _to_np(params_['velo_var_down'])

            f1 = _to_np(params_['burst_f1'])
            b1 = _to_np(params_['burst_B1'])
            gamma_ = _to_np(params_['gamma_mRNA_all'])
            mu_0 = f1 * b1 / gamma_
            var_0 = mu_0 * (b1 + 1)

            mu_up_f = _to_np(params_['mu_up_f'])
            var_up_f = _to_np(params_['var_up_f'])
            mu_down_up = _to_np(params_['mu_down_up'])
            var_down_up = _to_np(params_['var_down_up'])
            mu_down_f = _to_np(params_['mu_down_f'])
            var_down_f = _to_np(params_['var_down_f'])

            time_up = _to_np(params_['tau_up'])
            time_down = _to_np(params_['tau_down'])
            time_ss = _to_np(params_['time_ss'])
            loss_mu_full = _to_np(params_['loss_mu'])
            loss_std_full = _to_np(params_['loss_std'])

            mu_up, var_up = get_mu_var_time(mu_0, var_0, mu_up_f, var_up_f, gamma_, time_up)
            mu_down, var_down = get_mu_var_time(
                mu_down_up, var_down_up, mu_down_f, var_down_f, gamma_, time_down
            )

            probs_ = _to_np(params_['prob_state'])          # (chunk, G, 4)
            max_idx = np.argmax(probs_, axis=2)             # (chunk, G)

            n, G = mu_up.shape
            zeros = np.zeros((n, G), dtype=ACCUM_DTYPE)

            # Secondary (mu, var, velocity) for consistency/diagnostics only.
            # `max_idx` is still used for time_pred below.
            if state_reduction == "soft":
                mu_ = _mixture_by_state(probs_, (mu_up, mu_up_f, mu_down, mu_down_f))
                var_ = _mixture_by_state(probs_, (var_up, var_up_f, var_down, var_down_f))
                velo_mu_ = _mixture_by_state(probs_, (velo_mu_up, zeros, velo_mu_down, zeros))
                velo_var_ = _mixture_by_state(probs_, (velo_var_up, zeros, velo_var_down, zeros))
            else:
                mu_ = _select_by_state(max_idx, (mu_up, mu_up_f, mu_down, mu_down_f))
                var_ = _select_by_state(max_idx, (var_up, var_up_f, var_down, var_down_f))
                velo_mu_ = _select_by_state(max_idx, (velo_mu_up, zeros, velo_mu_down, zeros))
                velo_var_ = _select_by_state(max_idx, (velo_var_up, zeros, velo_var_down, zeros))

            time_pred = _select_by_state(
                max_idx,
                (time_up, time_ss, time_down + time_ss,
                 np.full((n, G), tmax, dtype=ACCUM_DTYPE)),
            )

            # ---- lazily allocate per-boot full-N buffers ----
            if buf is None:
                mu_moment_present = has_moment
                buf = {
                    "mu": np.empty((N, G), dtype=ACCUM_DTYPE),
                    "var": np.empty((N, G), dtype=ACCUM_DTYPE),
                    "velo_mu": np.empty((N, G), dtype=ACCUM_DTYPE),
                    "velo_var": np.empty((N, G), dtype=ACCUM_DTYPE),
                    "time_pred": np.empty((N, G), dtype=ACCUM_DTYPE),
                    "probs": np.empty((N, G, _N_STATES), dtype=ACCUM_DTYPE),
                    "loss_mu": np.empty((N, G), dtype=ACCUM_DTYPE),
                    "loss_std": np.empty((N, G), dtype=ACCUM_DTYPE),
                }
                if has_moment:
                    buf["mu_moment"] = np.empty((N, G), dtype=ACCUM_DTYPE)
                    buf["var_moment"] = np.empty((N, G), dtype=ACCUM_DTYPE)

            buf["mu"][idx] = mu_
            buf["var"][idx] = var_
            buf["velo_mu"][idx] = velo_mu_
            buf["velo_var"][idx] = velo_var_
            buf["time_pred"][idx] = time_pred
            buf["probs"][idx] = probs_
            buf["loss_mu"][idx] = loss_mu_full
            buf["loss_std"][idx] = loss_std_full
            if has_moment:
                buf["mu_moment"][idx] = mu_moment
                buf["var_moment"][idx] = var_moment

            # free the chunk
            del params_, probs_, max_idx, mu_up, var_up, mu_down, var_down
            del velo_mu_up, velo_var_up, velo_mu_down, velo_var_down
            del mu_0, var_0, f1, b1, gamma_
            torch.cuda.empty_cache()
            gc.collect()

        # ---- consistency needs the full graph: run once per boot ----
        velo_mu_consistency = compute_gene_velocity_consistency(adata_, buf["velo_mu"])
        velo_var_consistency = compute_gene_velocity_consistency(adata_, buf["velo_var"])

        # ---- accumulate over bootstraps ----
        if acc is None:
            acc = {k: v.astype(ACCUM_DTYPE, copy=True) for k, v in buf.items()}
            acc["velo_mu_consistency"] = np.asarray(velo_mu_consistency, dtype=ACCUM_DTYPE)
            acc["velo_var_consistency"] = np.asarray(velo_var_consistency, dtype=ACCUM_DTYPE)
        else:
            for k, v in buf.items():
                acc[k] += v
            acc["velo_mu_consistency"] += velo_mu_consistency
            acc["velo_var_consistency"] += velo_var_consistency

        del buf
        gc.collect()

    # ---- average ----
    for k in acc:
        acc[k] = acc[k] / nboot

    mu_moment_all = acc.get("mu_moment", None)
    var_moment_all = acc.get("var_moment", None)

    return (
        acc["mu"], acc["var"], acc["velo_mu"], acc["velo_var"],
        acc["time_pred"], acc["probs"], acc["loss_mu"], acc["loss_std"],
        mu_moment_all, var_moment_all,
        acc["velo_mu_consistency"], acc["velo_var_consistency"],
    )


# =============================================================================
# gene specific parameters (chunked; r2 accumulated over cell chunks)
# =============================================================================

def get_nosplicevelo_ll_params_gene(
    model_,
    adata_,
    nboot=10,
    prob_thresh=0.5,
    cell_batch_size: int = DEFAULT_CELL_BATCH,
    gpu_batch_size: Optional[int] = DEFAULT_GPU_BATCH,
    verbose: bool = True,
    state_reduction: str = "argmax_stable",
    extra_reductions=None,
    knn_weights=None,
    knn_n_iter: int = 1,
    temper_taus=(4.0,),
    extra_out: Optional[dict] = None,
):
    """Gene-specific velocity params, computed in cell batches.

    extra_reductions (optional; any of "argmax_knn", "tempered", "vote", "vote_knn"):
    "vote"/"vote_knn" are per-bootstrap argmax velocities averaged over bootstraps
    (vote_knn smooths each bootstrap's posterior over knn_weights first); velocity only.
    Additional
    reductions of the same per-state moments/velocities, written into the
    `extra_out` dict as '<mu|var|velo_mu|velo_var>_<name>' (N, G) arrays, with
    name 'argmax_knn' or 'tempered<tau>'. See state_reductions.py. They do not
    change the primary outputs. argmax_knn needs `knn_weights` (row-normalised
    (N, N) kNN matrix incl. self loops); `knn_n_iter` repeats the averaging.
    Ignored for state_reduction='argmax_perboot' (no per-state accumulation).

    R^2 quantities are sums over cells, so we accumulate their per-chunk
    contributions (d2, tss, counts) and combine at the end -- mathematically
    identical to the original full-matrix computation.

    state_reduction controls how the per-cell (mu, var, velocity) are built from
    the 4 kinetic states across the `nboot` bootstraps:

      "argmax_stable" (default): average each state's moments and the state
          probabilities across bootstraps FIRST, then pick one state per cell
          (argmax of the averaged posterior). Keeps the reconstruction ON the
          (mu, var) phase manifold -- mu/var/velocity all come from the same
          selected branch, so var is not pulled below the branch. Recommended.

      "soft": posterior-weighted mixture over states (using bootstrap-averaged
          per-state moments and probabilities). Smooth, but averages mu and var
          independently across a curved/bimodal locus, so points fall into the
          interior between branches (Jensen) and var is underestimated.

      "argmax_perboot": the original behaviour -- argmax state selected within
          each bootstrap, then the selected values averaged across bootstraps.
          When bootstraps disagree on the state this also averages across
          branches (off-manifold). Kept for reproducing prior results.

      "both": primary mu/var/velocity are argmax_stable (on-manifold), AND the
          soft posterior-mixture mu/var/velocities are additionally returned (as
          the last four tuple elements) -- e.g. use argmax_stable mu/var for the
          R2 / phase-portrait comparison and the soft mu/var/velocities for the
          downstream velocity graph / streams. Single inference pass.

    Returns four extra trailing elements
    (mu_soft, var_soft, velo_mu_soft, velo_var_soft); all None unless
    state_reduction == "both".
    """
    if state_reduction not in ("argmax_stable", "soft", "argmax_perboot", "both"):
        raise ValueError(f"unknown state_reduction={state_reduction!r}")
    mu_scvi_smooth = adata_.layers['mu_scvi_smooth']
    var_scvi_smooth = adata_.layers['var_scvi_smooth']
    if issparse(mu_scvi_smooth):
        mu_scvi_smooth = mu_scvi_smooth.toarray()
    if issparse(var_scvi_smooth):
        var_scvi_smooth = var_scvi_smooth.toarray()
    mu_scvi_smooth = np.asarray(mu_scvi_smooth, dtype=ACCUM_DTYPE)
    var_scvi_smooth = np.asarray(var_scvi_smooth, dtype=ACCUM_DTYPE)

    N = adata_.n_obs
    G = mu_scvi_smooth.shape[1]
    r2_lin_joint = np.zeros((N, G), dtype=ACCUM_DTYPE)

    acc = None
    # per-state moment/prob running sums across bootstraps (stable / soft modes)
    perstate = None

    # ---- optional per-bootstrap "vote" reductions (velocity only) ----
    # vote     : argmax state within each bootstrap -> that state's velocity -> mean over boots
    #            (= mixture weighted by how often each state wins across bootstraps)
    # vote_knn : same, but each bootstrap's posterior is first averaged over the kNN graph
    _vote_set = [r for r in (extra_reductions or []) if r in ("vote", "vote_knn")]
    _vote_acc, _boot_buf = None, None
    if _vote_set and extra_out is not None and state_reduction != "argmax_perboot":
        _vote_acc = {f"{q}_{r}": np.zeros((N, G), dtype=ACCUM_DTYPE)
                     for r in _vote_set for q in ("velo_mu", "velo_var")}
        if "vote_knn" in _vote_set:
            if knn_weights is None:
                raise ValueError("vote_knn needs knn_weights")
            _boot_buf = {"probs": np.zeros((N, G, _N_STATES), dtype=np.float32),
                         **{k: np.zeros((N, G), dtype=np.float32)
                            for k in ("vmu_up", "vmu_down", "vvar_up", "vvar_down")}}

    for b_ in range(nboot):
        if verbose:
            print(f"gene boot = {b_}")

        buf = None
        if state_reduction == "argmax_perboot":
            buf = {
                "mu": np.empty((N, G), dtype=ACCUM_DTYPE),
                "var": np.empty((N, G), dtype=ACCUM_DTYPE),
                "velo_mu": np.empty((N, G), dtype=ACCUM_DTYPE),
                "velo_var": np.empty((N, G), dtype=ACCUM_DTYPE),
            }
        # running r2 pieces (per gene)
        d2_up = np.zeros(G, dtype=np.float64)
        d2_down = np.zeros(G, dtype=np.float64)
        sum_mu_up = np.zeros(G, dtype=np.float64)
        sum_var_up = np.zeros(G, dtype=np.float64)
        sum_mu_down = np.zeros(G, dtype=np.float64)
        sum_var_down = np.zeros(G, dtype=np.float64)
        sumsq_mu_up = np.zeros(G, dtype=np.float64)
        sumsq_var_up = np.zeros(G, dtype=np.float64)
        sumsq_mu_down = np.zeros(G, dtype=np.float64)
        sumsq_var_down = np.zeros(G, dtype=np.float64)
        n_up = np.zeros(G, dtype=np.float64)
        n_down = np.zeros(G, dtype=np.float64)

        for idx, params_gene in _iter_cell_chunks(
            model_, "get_likelihood_parameters_gene_specific", N, cell_batch_size,
            gpu_batch_size=gpu_batch_size, adata=adata_,
        ):
            f1 = _to_np(params_gene['burst_f1'])
            b1 = _to_np(params_gene['burst_B1'])
            gamma_ = _to_np(params_gene['gamma_mRNA_all'])
            mu_0 = f1 * b1 / gamma_
            var_0 = mu_0 * (b1 + 1)

            mu_up_f = _to_np(params_gene['mu_up_f'])
            var_up_f = _to_np(params_gene['var_up_f'])
            mu_down_up = _to_np(params_gene['mu_down_up'])
            var_down_up = _to_np(params_gene['var_down_up'])
            mu_down_f = _to_np(params_gene['mu_down_f'])
            var_down_f = _to_np(params_gene['var_down_f'])
            time_ss = _to_np(params_gene['time_ss'])
            time_up = _to_np(params_gene['tau_up'])
            time_down = _to_np(params_gene['tau_down'])

            mu_up, var_up = get_mu_var_time(mu_0, var_0, mu_up_f, var_up_f, gamma_, time_up)
            mu_down, var_down = get_mu_var_time(
                mu_down_up, var_down_up, mu_down_f, var_down_f, gamma_, time_down
            )

            probs_ = _to_np(params_gene['prob_state'])
            max_idx = np.argmax(probs_, axis=2)

            n = len(idx)
            zeros = np.zeros((n, G), dtype=ACCUM_DTYPE)

            velo_mu_up = (mu_up_f - mu_up) * gamma_
            B_up_f = var_up_f / mu_up_f - 1
            velo_var_up = (mu_up_f * (2 * B_up_f + 1) + mu_up - 2 * var_up) * gamma_
            velo_mu_down = (mu_down_f - mu_down) * gamma_
            B_down_f = var_down_f / mu_down_f - 1
            velo_var_down = (mu_down_f * (2 * B_down_f + 1) + mu_down - 2 * var_down) * gamma_

            # Build the reconstructed (mu, var, velocity). `max_idx` is still used
            # below for the per-state r2 masks (which need a hard assignment).
            if state_reduction == "argmax_perboot":
                # original: select argmax state within this bootstrap
                buf["mu"][idx] = _select_by_state(max_idx, (mu_up, mu_up_f, mu_down, mu_down_f))
                buf["var"][idx] = _select_by_state(max_idx, (var_up, var_up_f, var_down, var_down_f))
                buf["velo_mu"][idx] = _select_by_state(max_idx, (velo_mu_up, zeros, velo_mu_down, zeros))
                buf["velo_var"][idx] = _select_by_state(max_idx, (velo_var_up, zeros, velo_var_down, zeros))
            else:
                # accumulate per-state moments + probs across bootstraps; the
                # selection / mixture is done ONCE after all boots (on-manifold).
                if perstate is None:
                    perstate = {k: np.zeros((N, G), dtype=ACCUM_DTYPE) for k in (
                        "mu_up", "mu_up_f", "mu_down", "mu_down_f",
                        "var_up", "var_up_f", "var_down", "var_down_f",
                        "vmu_up", "vmu_down", "vvar_up", "vvar_down")}
                    perstate["probs"] = np.zeros((N, G, _N_STATES), dtype=ACCUM_DTYPE)
                perstate["mu_up"][idx] += mu_up
                perstate["mu_up_f"][idx] += mu_up_f
                perstate["mu_down"][idx] += mu_down
                perstate["mu_down_f"][idx] += mu_down_f
                perstate["var_up"][idx] += var_up
                perstate["var_up_f"][idx] += var_up_f
                perstate["var_down"][idx] += var_down
                perstate["var_down_f"][idx] += var_down_f
                perstate["vmu_up"][idx] += velo_mu_up
                perstate["vmu_down"][idx] += velo_mu_down
                perstate["vvar_up"][idx] += velo_var_up
                perstate["vvar_down"][idx] += velo_var_down
                perstate["probs"][idx] += probs_

            if _vote_acc is not None:
                if "vote" in _vote_set:
                    _vote_acc["velo_mu_vote"][idx] += _select_by_state(
                        max_idx, (velo_mu_up, zeros, velo_mu_down, zeros))
                    _vote_acc["velo_var_vote"][idx] += _select_by_state(
                        max_idx, (velo_var_up, zeros, velo_var_down, zeros))
                if _boot_buf is not None:
                    _boot_buf["probs"][idx] = probs_
                    _boot_buf["vmu_up"][idx] = velo_mu_up
                    _boot_buf["vmu_down"][idx] = velo_mu_down
                    _boot_buf["vvar_up"][idx] = velo_var_up
                    _boot_buf["vvar_down"][idx] = velo_var_down

            # ---- r2 pieces (sums over cells within this chunk) ----
            ms = mu_scvi_smooth[idx]
            vs = var_scvi_smooth[idx]

            mask_up = (max_idx == 0).astype(np.float64)
            res_up = (ms - mu_up) ** 2 + (vs - var_up) ** 2
            d2_up += np.sum(res_up * mask_up, axis=0)
            sum_mu_up += np.sum(ms * mask_up, axis=0)
            sum_var_up += np.sum(vs * mask_up, axis=0)
            sumsq_mu_up += np.sum((ms ** 2) * mask_up, axis=0)
            sumsq_var_up += np.sum((vs ** 2) * mask_up, axis=0)
            n_up += np.sum(mask_up, axis=0)

            mask_down = (max_idx == 2).astype(np.float64)
            res_down = (ms - mu_down) ** 2 + (vs - var_down) ** 2
            d2_down += np.sum(res_down * mask_down, axis=0)
            sum_mu_down += np.sum(ms * mask_down, axis=0)
            sum_var_down += np.sum(vs * mask_down, axis=0)
            sumsq_mu_down += np.sum((ms ** 2) * mask_down, axis=0)
            sumsq_var_down += np.sum((vs ** 2) * mask_down, axis=0)
            n_down += np.sum(mask_down, axis=0)

            del params_gene, probs_, max_idx
            torch.cuda.empty_cache()
            gc.collect()

        if _boot_buf is not None:
            from state_reductions import smooth_posterior as _smooth_posterior
            _st = np.argmax(_smooth_posterior(_boot_buf["probs"], knn_weights, n_iter=knn_n_iter), axis=2)
            _z = np.zeros((N, G), dtype=np.float32)
            _vote_acc["velo_mu_vote_knn"] += _select_by_state(
                _st, (_boot_buf["vmu_up"], _z, _boot_buf["vmu_down"], _z))
            _vote_acc["velo_var_vote_knn"] += _select_by_state(
                _st, (_boot_buf["vvar_up"], _z, _boot_buf["vvar_down"], _z))
            del _st, _z

        # ---- finalize r2 for this boot ----
        # tss = sum (x - mean)^2 = sumsq - sum^2/n, computed from running sums
        mean_mu_up = sum_mu_up / (n_up + 1e-8)
        mean_var_up = sum_var_up / (n_up + 1e-8)
        tss_up = (sumsq_mu_up - n_up * mean_mu_up ** 2) + (sumsq_var_up - n_up * mean_var_up ** 2) + 1e-8
        r2_model_up = 1 - d2_up / tss_up

        mean_mu_down = sum_mu_down / (n_down + 1e-8)
        mean_var_down = sum_var_down / (n_down + 1e-8)
        tss_down = (sumsq_mu_down - n_down * mean_mu_down ** 2) + (sumsq_var_down - n_down * mean_var_down ** 2) + 1e-8
        r2_model_down = 1 - d2_down / tss_down

        r2_lin_up = np.zeros(G, dtype=ACCUM_DTYPE)
        r2_lin_down = np.zeros(G, dtype=ACCUM_DTYPE)
        n_counts_dict = {'n_counts_up': n_up.copy(), 'n_counts_down': n_down.copy()}

        boot = {
            "r2_model_up": r2_model_up.astype(ACCUM_DTYPE),
            "r2_lin_up": r2_lin_up,
            "r2_model_down": r2_model_down.astype(ACCUM_DTYPE),
            "r2_lin_down": r2_lin_down,
        }
        if state_reduction == "argmax_perboot":
            boot["mu"] = buf["mu"]; boot["var"] = buf["var"]
            boot["velo_mu"] = buf["velo_mu"]; boot["velo_var"] = buf["velo_var"]
        if acc is None:
            acc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in boot.items()}
        else:
            for k in acc:
                acc[k] += boot[k]
        del boot
        if buf is not None:
            del buf
        gc.collect()

    for k in acc:
        acc[k] = acc[k] / nboot

    # ---- build the reconstructed (mu, var, velocity) ----
    mu_soft = None
    var_soft = None
    velo_mu_soft = None
    velo_var_soft = None
    if state_reduction == "argmax_perboot":
        mu_out, var_out = acc["mu"], acc["var"]
        velo_mu_out, velo_var_out = acc["velo_mu"], acc["velo_var"]
    else:
        inv = 1.0 / nboot
        muA = (perstate["mu_up"] * inv, perstate["mu_up_f"] * inv,
               perstate["mu_down"] * inv, perstate["mu_down_f"] * inv)
        varA = (perstate["var_up"] * inv, perstate["var_up_f"] * inv,
                perstate["var_down"] * inv, perstate["var_down_f"] * inv)
        z = np.zeros((N, G), dtype=ACCUM_DTYPE)
        vmuA = (perstate["vmu_up"] * inv, z, perstate["vmu_down"] * inv, z)
        vvarA = (perstate["vvar_up"] * inv, z, perstate["vvar_down"] * inv, z)
        probA = perstate["probs"] * inv
        if state_reduction == "soft":
            mu_out = _mixture_by_state(probA, muA)
            var_out = _mixture_by_state(probA, varA)
            velo_mu_out = _mixture_by_state(probA, vmuA)
            velo_var_out = _mixture_by_state(probA, vvarA)
        else:  # "argmax_stable" or "both": on-manifold moments + velocity
            state = np.argmax(probA, axis=2)
            mu_out = _select_by_state(state, muA)
            var_out = _select_by_state(state, varA)
            velo_mu_out = _select_by_state(state, vmuA)
            velo_var_out = _select_by_state(state, vvarA)
            if state_reduction == "both":
                # additionally expose the soft posterior-mixture mu/var/velocities
                mu_soft = _mixture_by_state(probA, muA)
                var_soft = _mixture_by_state(probA, varA)
                velo_mu_soft = _mixture_by_state(probA, vmuA)
                velo_var_soft = _mixture_by_state(probA, vvarA)

    if extra_reductions and extra_out is not None:
        if state_reduction == "argmax_perboot":
            print("[ll_params] extra_reductions ignored for state_reduction='argmax_perboot'")
        else:
            _posthoc = [r for r in extra_reductions if r not in ("vote", "vote_knn")]
            if _posthoc:
                from state_reductions import extra_reductions as _extra_reductions
                extra_out.update(_extra_reductions(
                    probA, muA, varA, vmuA, vvarA, _posthoc, W=knn_weights,
                    n_iter=knn_n_iter, taus=temper_taus, verbose=verbose))
            if _vote_acc is not None:
                for _k, _v in _vote_acc.items():
                    extra_out[_k] = _v / nboot
                if verbose:
                    print(f"[ll_params] vote reductions over {nboot} bootstraps: {sorted(_vote_acc)}")

    delta_r2_var = np.zeros(G, dtype=ACCUM_DTYPE)
    return (
        mu_out, var_out, velo_mu_out, velo_var_out,
        acc["r2_model_up"], acc["r2_lin_up"], acc["r2_model_down"], acc["r2_lin_down"],
        n_counts_dict, delta_r2_var, r2_lin_joint,
        mu_soft, var_soft, velo_mu_soft, velo_var_soft,
    )


# =============================================================================
# bootstrap-averaged gene-specific steady-state / up-down parameters
# =============================================================================

def get_nosplicevelo_ll_params_muVarUp_gene(
    model_,
    adata_=None,
    nboot=10,
    prob_thresh=0.5,
    cell_batch_size: int = DEFAULT_CELL_BATCH,
    gpu_batch_size: Optional[int] = DEFAULT_GPU_BATCH,
    verbose: bool = True,
):
    """Bootstrap-averaged gene-specific (mu_up_f, var_up_f, mu_down_up, var_down_up).

    Same result as the original `get_nosplicevelo_ll_params_muVarUp_gene`
    (mean over `nboot` draws of `get_likelihood_parameters_gene_specific`), but
    each draw is pulled in cell chunks so the model never assembles all cells at
    once. `prob_thresh` is accepted for signature compatibility (unused, as in
    the original).

    Parameters
    ----------
    adata_ :
        Needed only for cell-chunking (n_obs + optional `adata=` subsetting).
        If None, falls back to a single whole-dataset model call per boot
        (matches the original behavior exactly).

    Returns
    -------
    (mu_0, var_0, mu_up_f, var_up_f, mu_down_up, var_down_up, mu_down_f,
     var_down_f) : np.ndarray each (N, G). mu_0/var_0 are derived from the
    bootstrap-averaged burst_f1/burst_B1/gamma_mRNA_all.
    """
    # Fetch every key needed to build the 8-tuple: the four *_up/down_up terms,
    # the two down_f terms, and the burst/gamma params used to derive mu_0/var_0.
    wanted = (
        "mu_up_f", "var_up_f", "mu_down_up", "var_down_up",
        "mu_down_f", "var_down_f",
        "burst_f1", "burst_B1", "gamma_mRNA_all",
    )
    acc = None

    for b_ in range(nboot):
        if verbose:
            print(f"gene boot = {b_}")

        # ---- one bootstrap draw, optionally chunked over cells ----
        if adata_ is None:
            # Original path: whole dataset in one call.
            params_gene = _infer_ll_params(
                model_, "get_likelihood_parameters_gene_specific",
                batch_size=gpu_batch_size,
            )
            draw = {k: _to_np(params_gene[k]) for k in wanted}
            del params_gene
        else:
            N = adata_.n_obs
            draw = None
            for idx, params_gene in _iter_cell_chunks(
                model_, "get_likelihood_parameters_gene_specific", N, cell_batch_size,
                gpu_batch_size=gpu_batch_size, adata=adata_,
            ):
                if draw is None:
                    G = np.asarray(params_gene[wanted[0]]).shape[1]
                    draw = {k: np.empty((N, G), dtype=ACCUM_DTYPE) for k in wanted}
                for k in wanted:
                    draw[k][idx] = _to_np(params_gene[k])
                del params_gene
                torch.cuda.empty_cache()
                gc.collect()

        # ---- accumulate over bootstraps ----
        if acc is None:
            acc = {k: v.copy() for k, v in draw.items()}
        else:
            for k in wanted:
                acc[k] += draw[k]
        del draw
        gc.collect()

    for k in wanted:
        acc[k] = acc[k] / nboot
        
    f1_gene = acc['burst_f1'].copy()
    b1_gene = acc['burst_B1'].copy()
    gamma_gene = acc['gamma_mRNA_all'].copy()
        
    mu_0_gene = f1_gene * b1_gene / gamma_gene
    var_0_gene = mu_0_gene * (b1_gene + 1)
    acc['mu_0'] = mu_0_gene
    acc['var_0'] = var_0_gene

    return acc['mu_0'], acc['var_0'], acc["mu_up_f"], acc["var_up_f"], acc["mu_down_up"], acc["var_down_up"], \
        acc["mu_down_f"], acc["var_down_f"]


# =============================================================================
# top-level entry point (same signature/return as the original)
# =============================================================================

def get_nosplicevelo_ll_params(
    model_,
    model_0,
    adata_,
    nboot=10,
    prob_thresh=0.5,
    basis="umap",
    cell_batch_size: int = DEFAULT_CELL_BATCH,
    gpu_batch_size: Optional[int] = DEFAULT_GPU_BATCH,
    verbose: bool = True,
    state_reduction: str = "argmax_stable",
    extra_reductions=None,
    knn_weights=None,
    knn_n_iter: int = 1,
    temper_taus=(4.0,),
):
    """Drop-in replacement for the original, with cell-batched inference.

    extra_reductions / knn_weights / knn_n_iter / temper_taus: see
    get_nosplicevelo_ll_params_gene; results are added to params_all as
    '<key>_all_gene' (e.g. 'velo_mu_argmax_knn_all_gene', 'velo_mu_tempered4_all_gene').

    state_reduction: "argmax_stable" (default, on-manifold) | "soft" |
    "argmax_perboot" (original). See get_nosplicevelo_ll_params_gene.

    New keywords:
        cell_batch_size : cells per outer chunk / model call (bounds host RAM).
        gpu_batch_size  : inner dataloader batch_size for the GPU forward
                          (bounds GPU RAM). None -> scvi.settings.batch_size.

    Returns the same `params_all` dict as the original.
    """
    if verbose:
        describe_model_support(model_, "get_likelihood_parameters_new")

    _extra = {}
    (mu_all_gene, var_all_gene, velo_mu_all_gene, velo_var_all_gene,
     r2_model_up_all, r2_lin_up_all, r2_model_down_all, r2_lin_down_all,
     n_counts_dict, delta_r2_var, r2_lin_gene_all,
     mu_soft_all_gene, var_soft_all_gene,
     velo_mu_soft_all_gene, velo_var_soft_all_gene) = get_nosplicevelo_ll_params_gene(
        model_, adata_, prob_thresh=prob_thresh, nboot=nboot,
        cell_batch_size=cell_batch_size, gpu_batch_size=gpu_batch_size, verbose=verbose,
        state_reduction=state_reduction,
        extra_reductions=extra_reductions, knn_weights=knn_weights, knn_n_iter=knn_n_iter,
        temper_taus=temper_taus, extra_out=_extra,
    )

    (mu_all, var_all, velo_mu_all, velo_var_all, time_pred_all, probs_all,
     loss_mu, loss_std, mu_moment_all, var_moment_all,
     velo_mu_consistency_all, velo_var_consistency_all) = get_nosplicevelo_ll_params_geneCell(
        model_, model_0, adata_, prob_thresh=prob_thresh, nboot=nboot, basis=basis,
        cell_batch_size=cell_batch_size, gpu_batch_size=gpu_batch_size, verbose=verbose,
        state_reduction=state_reduction,
    )

    params_all = {
        'mu_all_gene': mu_all_gene,
        'var_all_gene': var_all_gene,
        'velo_mu_all_gene': velo_mu_all_gene,
        'velo_var_all_gene': velo_var_all_gene,
        'r2_model_up_all': r2_model_up_all,
        'r2_lin_up_all': r2_lin_up_all,
        'r2_model_down_all': r2_model_down_all,
        'r2_lin_down_all': r2_lin_down_all,
        'n_counts_dict': n_counts_dict,
        'delta_r2_var': delta_r2_var,
        'r2_lin_gene_all': r2_lin_gene_all,
        'velo_mu_consistency_all': velo_mu_consistency_all,
        'velo_var_consistency_all': velo_var_consistency_all,
        'mu_all': mu_all,
        'var_all': var_all,
        'velo_mu_all': velo_mu_all,
        'velo_var_all': velo_var_all,
        'time_pred_all': time_pred_all,
        'probs_all': probs_all,
        'loss_mu': loss_mu,
        'loss_std': loss_std,
    }
    if mu_moment_all is not None:
        params_all['mu_moment_all'] = mu_moment_all
        params_all['var_moment_all'] = var_moment_all
    if velo_mu_soft_all_gene is not None:
        params_all['mu_soft_all_gene'] = mu_soft_all_gene
        params_all['var_soft_all_gene'] = var_soft_all_gene
        params_all['velo_mu_soft_all_gene'] = velo_mu_soft_all_gene
        params_all['velo_var_soft_all_gene'] = velo_var_soft_all_gene
    for _k, _v in _extra.items():
        params_all[f"{_k}_all_gene"] = _v

    return params_all


# =============================================================================
# MODEL_PATCH_EXAMPLE
# =============================================================================
MODEL_PATCH_EXAMPLE = r'''
If get_likelihood_parameters_new does NOT accept batch_size/indices/adata,
add batching inside the model class (scvi-tools style). This is the cleanest
fix and keeps all heavy work on the GPU in bounded batches:

    @torch.inference_mode()
    def get_likelihood_parameters_new(self, adata=None, indices=None,
                                      batch_size=None):
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices,
            batch_size=batch_size or settings.batch_size,
        )
        out = {}  # key -> list of per-batch numpy arrays
        for tensors in scdl:
            inference_out = self.module.inference(**self.module._get_inference_input(tensors))
            generative_out = self.module.generative(**self.module._get_generative_input(tensors, inference_out))
            batch = self._compute_ll_params(inference_out, generative_out)  # your existing math
            for k, v in batch.items():
                out.setdefault(k, []).append(v.detach().cpu().numpy())
        return {k: np.concatenate(v, axis=0) for k, v in out.items()}

Then this module's Tier-A path handles everything with a single call and a
sensible batch_size.
'''
