"""
score_gene_fit_v2 -- per-gene fit quality with NO 2D geometric R2.

Everything is a PER-AXIS R2 (mu and var scored separately, then min-combined),
computed PER BRANCH (cells split up/down by argmax branch posterior; a branch is
only scored when it has >= branch_min_cells cells), then aggregated over the two
branches (default: max). Two nulls are always computed:

  fit_r2_branch_meanNull : per-axis R2 vs the per-branch flat MEAN
  fit_r2_branch_lineNull : per-axis R2 vs the per-branch TLS-LINE prediction
                           (mu_tls, var_tls from the orthogonal projection)

mu and var R2 are kept SEPARATELY (of the selected branch) so genes that are bad
for mu, bad for var, or both can be told apart; a mu-vs-var scatter is produced
per null. There is NO complete-data (whole-cloud) fit metric -- branch-specific
only. Reliability is reported for BOTH nulls separately
(velocity_reliable_meanNull / velocity_reliable_lineNull), so there is no --null
flag.

Uses the four noSpliceVelo layers:
  observed : mu_scvi_smooth , var_scvi_smooth
  fitted   : mu_fit         , var_fit        (argmax_stable recommended)
and the state posterior (prob_state_avg_nosplicevelo.npy, [N,G,S]).
"""

import os
import sys
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# reuse shared, tested helpers from v1
from score_gene_fit import (branch_posterior, auto_conf_threshold,
                            report_layer_health, _r2_meannull,
                            plot_branch_r2_scatter)


# ---------------------------------------------------------------------------
# per-axis R2 within a branch mask (mean null and TLS-line null)
# ---------------------------------------------------------------------------

def _meannull_axes(mu, var, mf, vf, mask):
    """Per-axis mean-null R2 (r2_mu, r2_var) within `mask` [N,G]."""
    return _r2_meannull(mu, mf, mask), _r2_meannull(var, vf, mask)


def _linenull_axes(mu, var, mf, vf, mask, eps=1e-12):
    """Per-axis R2 vs the TLS-line prediction (r2_mu, r2_var) within `mask`.
    Fit a TLS line to the masked (mu,var) cloud, take its per-cell predicted mu
    and var (orthogonal projection), and score each axis:
        R2_mu = 1 - sum(mu_obs-mu_fit)^2 / sum(mu_obs-mu_tls)^2   (and var)."""
    mu = np.where(mask, mu, np.nan); var = np.where(mask, var, np.nan)
    mf = np.where(mask, mf, np.nan); vf = np.where(mask, vf, np.nan)
    s_mu = np.nanstd(mu, 0); s_mu = np.where(s_mu < eps, eps, s_mu)
    s_var = np.nanstd(var, 0); s_var = np.where(s_var < eps, eps, s_var)
    x = mu / s_mu; y = var / s_var
    xm = np.nanmean(x, 0); ym = np.nanmean(y, 0)
    xc = x - xm; yc = y - ym
    Sxx = np.nansum(xc ** 2, 0); Syy = np.nansum(yc ** 2, 0); Sxy = np.nansum(xc * yc, 0)
    th = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    ux = np.cos(th); uy = np.sin(th)
    t = xc * ux + yc * uy
    mu_tls = (t * ux + xm) * s_mu
    var_tls = (t * uy + ym) * s_var
    r2_mu = 1.0 - np.nansum((mu - mf) ** 2, 0) / (np.nansum((mu - mu_tls) ** 2, 0) + eps)
    r2_var = 1.0 - np.nansum((var - vf) ** 2, 0) / (np.nansum((var - var_tls) ** 2, 0) + eps)
    return r2_mu, r2_var


def _agg_axes(c_up, c_dn, mu_up, mu_dn, var_up, var_dn, n_up, n_dn, min_cells, agg):
    """Gate branches, aggregate the combined score AND carry the selected
    branch's mu/var R2 with it. Returns (r2, r2_mu, r2_var, c_up_gated,
    c_dn_gated). For max/min the mu/var come from the picked branch; for
    weighted they are the cell-weighted mean."""
    gu = n_up >= min_cells; gd = n_dn >= min_cells
    cu = np.where(gu, c_up, np.nan); cd = np.where(gd, c_dn, np.nan)
    mu_u = np.where(gu, mu_up, np.nan); mu_d = np.where(gd, mu_dn, np.nan)
    vu = np.where(gu, var_up, np.nan); vd = np.where(gd, var_dn, np.nan)
    both_nan = np.isnan(cu) & np.isnan(cd)
    if agg in ("max", "min"):
        if agg == "max":
            pick_up = np.nan_to_num(cu, nan=-np.inf) >= np.nan_to_num(cd, nan=-np.inf)
        else:
            pick_up = np.nan_to_num(cu, nan=np.inf) <= np.nan_to_num(cd, nan=np.inf)
        r2 = np.where(pick_up, cu, cd)
        r2_mu = np.where(pick_up, mu_u, mu_d)
        r2_var = np.where(pick_up, vu, vd)
    else:  # weighted
        wu = np.where(gu, n_up, 0.0); wd = np.where(gd, n_dn, 0.0); ws = wu + wd + 1e-12
        r2 = (np.nan_to_num(cu) * wu + np.nan_to_num(cd) * wd) / ws
        r2_mu = (np.nan_to_num(mu_u) * wu + np.nan_to_num(mu_d) * wd) / ws
        r2_var = (np.nan_to_num(vu) * wu + np.nan_to_num(vd) * wd) / ws
    r2 = np.where(both_nan, np.nan, r2)
    r2_mu = np.where(both_nan, np.nan, r2_mu)
    r2_var = np.where(both_nan, np.nan, r2_var)
    return r2, r2_mu, r2_var, cu, cd


def perbranch_r2(mu_obs, var_obs, mu_fit, var_fit, prob_state, null="mean",
                 up_states=(0, 1), down_states=(2, 3), min_cells=30, agg="max",
                 keep_mask=None, combine="min"):
    """Per-axis R2 per branch (gated at min_cells), aggregated over up/down.
    null='mean' or 'line'. Returns dict with:
      r2       aggregated combined (min of mu/var) score
      r2_mu    mu R2 of the selected branch     r2_var  var R2 of selected branch
      r2_up    per-branch combined (up, gated)  r2_down per-branch combined (down)
      n_up, n_down"""
    p_up, p_down, _, _ = branch_posterior(prob_state, up_states, down_states)
    up_mask = p_up >= p_down
    dn_mask = ~up_mask
    if keep_mask is not None:
        up_mask = up_mask & keep_mask
        dn_mask = dn_mask & keep_mask
    axes = _meannull_axes if null == "mean" else _linenull_axes
    comb = np.minimum if combine == "min" else (lambda a, b: 0.5 * (a + b))

    def branch(mask):
        r2_mu, r2_var = axes(mu_obs, var_obs, mu_fit, var_fit, mask)
        return r2_mu, r2_var, comb(r2_mu, r2_var), mask.sum(0)

    mu_u, var_u, c_u, n_u = branch(up_mask)
    mu_d, var_d, c_d, n_d = branch(dn_mask)
    r2, r2_mu, r2_var, c_up_g, c_dn_g = _agg_axes(
        c_u, c_d, mu_u, mu_d, var_u, var_d, n_u, n_d, min_cells, agg)
    return dict(r2=r2, r2_mu=r2_mu, r2_var=r2_var, r2_up=c_up_g, r2_down=c_dn_g,
                n_up=n_u, n_down=n_d)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def confidence_report(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                      conf_thresh=0.9, up_states=(0, 1), down_states=(2, 3),
                      max_drop_pct=20.0, combine="min", branch_min_cells=30,
                      branch_agg="max"):
    _, _, _, bc = branch_posterior(prob_state, up_states, down_states)
    keep = bc >= conf_thresh
    out = {}
    n_ref = None
    for null, tag in (("mean", "meanNull"), ("line", "lineNull")):
        pb = perbranch_r2(mu_obs, var_obs, mu_fit, var_fit, prob_state, null,
                          up_states, down_states, branch_min_cells, branch_agg,
                          None, combine)
        pbc = perbranch_r2(mu_obs, var_obs, mu_fit, var_fit, prob_state, null,
                           up_states, down_states, branch_min_cells, branch_agg,
                           keep, combine)
        out[f"fit_r2_branch_{tag}"] = pb["r2"]
        out[f"fit_r2_branch_{tag}_mu"] = pb["r2_mu"]
        out[f"fit_r2_branch_{tag}_var"] = pb["r2_var"]
        out[f"fit_r2_branch_{tag}_up"] = pb["r2_up"]
        out[f"fit_r2_branch_{tag}_down"] = pb["r2_down"]
        out[f"fit_r2_branch_{tag}_conf"] = pbc["r2"]
        out[f"fit_r2_branch_{tag}_mu_conf"] = pbc["r2_mu"]
        out[f"fit_r2_branch_{tag}_var_conf"] = pbc["r2_var"]
        out[f"fit_r2_branch_{tag}_up_conf"] = pbc["r2_up"]
        out[f"fit_r2_branch_{tag}_down_conf"] = pbc["r2_down"]
        if n_ref is None:
            n_ref = (pb["n_up"], pb["n_down"], pbc["n_up"], pbc["n_down"])
    out["n_up"], out["n_down"], out["n_up_conf"], out["n_down_conf"] = n_ref
    out["pct_cells_retained"] = 100.0 * keep.mean(0)
    out["high_cell_loss"] = (100.0 - out["pct_cells_retained"]) > max_drop_pct
    return out


def print_confidence_report(rep, conf_thresh, max_drop_pct):
    def med(x): return float(np.nanmedian(x))
    print(f"\n=== per-branch per-axis R2 (mu & var separate, min-combined; "
          f"branch gated at min_cells; aggregated = max over up/down) ===")
    print(f"{'metric (median over genes)':34s}{'all':>10}{'confident':>12}")
    for tag in ("meanNull", "lineNull"):
        print(f"{'fit_r2_branch_'+tag:34s}"
              f"{med(rep['fit_r2_branch_'+tag]):>10.3f}"
              f"{med(rep['fit_r2_branch_'+tag+'_conf']):>12.3f}")
        print(f"{'   mu / var (selected branch)':34s}"
              f"{med(rep['fit_r2_branch_'+tag+'_mu']):>10.3f}"
              f"{med(rep['fit_r2_branch_'+tag+'_var']):>12.3f}")
    pr = rep['pct_cells_retained']
    print(f"\ncells retained after filter: median {np.median(pr):.1f}%  "
          f"min {np.min(pr):.1f}%")
    print(f"genes flagged high_cell_loss (> {max_drop_pct:.0f}% dropped): "
          f"{int(rep['high_cell_loss'].sum())} / {len(pr)}")


# ---------------------------------------------------------------------------
# meanNull vs lineNull comparison scatter
# ---------------------------------------------------------------------------

def plot_null_comparison(x, y, out_png, xlabel="fit_r2_branch_meanNull",
                         ylabel="fit_r2_branch_lineNull", clip_min=-1.0,
                         title="mean-null vs line-null per-branch R2"):
    """Scatter of the aggregated per-branch R2 under the two nulls, one point per
    gene, over a KDE density. Genes with no valid branch (NaN) are dropped;
    remaining values are clamped at clip_min."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    xs = np.maximum(x[ok], clip_min); ys = np.maximum(y[ok], clip_min)
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    if xs.size >= 10:
        try:
            import seaborn as sns
            sns.kdeplot(x=xs, y=ys, fill=True, cmap="Blues", thresh=0.05,
                        levels=15, ax=ax)
        except Exception:
            hb = ax.hexbin(xs, ys, gridsize=40, cmap="Blues", mincnt=1)
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="genes")
    ax.scatter(xs, ys, s=9, c="#222222", alpha=0.35, linewidths=0)
    ax.axline((0, 0), slope=1, ls="--", c="gray", lw=1)
    ax.axhline(0, c="gray", lw=0.6, ls=":"); ax.axvline(0, c="gray", lw=0.6, ls=":")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path):
        self.file = open(path, "w"); self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s); self.file.write(s)

    def flush(self):
        self.stdout.flush(); self.file.flush()

    def close(self):
        self.file.flush(); self.file.close()


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# h5ad driver
# ---------------------------------------------------------------------------

def run_on_h5ad(path, mu_obs_layer="mu_scvi_smooth", var_obs_layer="var_scvi_smooth",
                mu_fit_layer="mu_fit", var_fit_layer="var_fit", combine="min",
                write_h5ad=False, out_h5ad=None, out_csv=None, log_file=None,
                out_dir=None, diagnose_only=False, prob_state=None, conf_thresh=0.9,
                up_states=(0, 1), down_states=(2, 3), max_drop_pct=20.0,
                target_error=None, branch_min_cells=30, branch_agg="max",
                thresh_meannull=0.5, thresh_linenull=0.0):
    out_dir = out_dir or (os.path.dirname(path) or ".")
    os.makedirs(out_dir, exist_ok=True)
    if log_file is None:
        log_file = os.path.join(out_dir, f"gene_fit_v2_{time.strftime('%Y%m%d_%H%M%S')}.log")
    tee = _Tee(log_file); old = sys.stdout; sys.stdout = tee
    try:
        import anndata as ad
        import pandas as pd
        print(f"[{_now()}] score_gene_fit_v2 start\n  h5ad: {os.path.abspath(path)}\n"
              f"  log : {os.path.abspath(log_file)}\n  out_dir: {os.path.abspath(out_dir)}")
        t0 = time.time()
        adata = ad.read_h5ad(path)
        print(f"  loaded {adata.n_obs} cells x {adata.n_vars} genes ({time.time()-t0:.1f}s)")

        def layer(n):
            if n not in adata.layers:
                raise KeyError(f"layer '{n}' not in {path}; have {list(adata.layers)}")
            return np.asarray(adata.layers[n])

        mo, vo = layer(mu_obs_layer), layer(var_obs_layer)
        mf, vf = layer(mu_fit_layer), layer(var_fit_layer)
        report_layer_health(mo, vo, mf, vf)
        if diagnose_only:
            print("\n[--diagnose] health only."); return None

        # branch-specific only (no complete-data fit_r2)
        csv_cols = {"gene": np.asarray(adata.var_names)}

        # prob_state
        prob = None
        try:
            if prob_state:
                prob = np.load(prob_state)
            else:
                in_dir = os.path.dirname(path) or "."
                for fn in ("prob_state_avg_nosplicevelo.npy", "prob_state_avg.npy"):
                    c = os.path.join(in_dir, fn)
                    if os.path.exists(c):
                        prob = np.load(c); print(f"  prob_state: {c}"); break
        except Exception as e:
            print(f"  [warn] prob_state load: {e}")

        if prob is None or prob.ndim != 3 or prob.shape[0] != adata.n_obs:
            print("  [note] no valid prob_state -> cannot compute branch report.")
        else:
            if target_error is not None:
                _, _, _, bc = branch_posterior(prob, up_states, down_states)
                conf_thresh, ret, err = auto_conf_threshold(bc, target_error=target_error)
                print(f"  [auto] target_error<={target_error} -> conf_thresh="
                      f"{conf_thresh:.3f} (retain {ret*100:.0f}%, err {err:.3f})")
            print(f"  prob_state shape {prob.shape}  up={up_states} down={down_states}"
                  f"  conf_thresh={conf_thresh:.3f}  agg={branch_agg}")
            rep = confidence_report(mo, vo, mf, vf, prob, conf_thresh=conf_thresh,
                                    up_states=up_states, down_states=down_states,
                                    max_drop_pct=max_drop_pct, combine=combine,
                                    branch_min_cells=branch_min_cells,
                                    branch_agg=branch_agg)
            print_confidence_report(rep, conf_thresh, max_drop_pct)

            # reliability for each null separately (no --null flag)
            rep["velocity_reliable_meanNull"] = rep["fit_r2_branch_meanNull"] >= thresh_meannull
            rep["velocity_reliable_lineNull"] = rep["fit_r2_branch_lineNull"] >= thresh_linenull
            print(f"\nreliable (meanNull>={thresh_meannull}): "
                  f"{int(np.nansum(rep['velocity_reliable_meanNull']))}   "
                  f"reliable (lineNull>={thresh_linenull}): "
                  f"{int(np.nansum(rep['velocity_reliable_lineNull']))}")

            for k, v in rep.items():
                adata.var[k] = v
                csv_cols[k] = v

            # --- scatter plots (all clamp at -1) ---
            try:
                paths = []
                for tag in ("meanNull", "lineNull"):
                    # up vs down (combined score)
                    paths.append(plot_branch_r2_scatter(
                        rep[f"fit_r2_branch_{tag}_up"], rep[f"fit_r2_branch_{tag}_down"],
                        rep["n_up"], rep["n_down"], branch_min_cells,
                        os.path.join(out_dir, f"branch_fit_r2_{tag}_scatter.png"),
                        title=f"per-branch {tag} R2  up vs down (all cells)", clip_min=-1.0))
                    paths.append(plot_branch_r2_scatter(
                        rep[f"fit_r2_branch_{tag}_up_conf"], rep[f"fit_r2_branch_{tag}_down_conf"],
                        rep["n_up_conf"], rep["n_down_conf"], branch_min_cells,
                        os.path.join(out_dir, f"branch_fit_r2_{tag}_scatter_conf.png"),
                        title=f"per-branch {tag} R2  up vs down (confident)", clip_min=-1.0))
                    # mu vs var (selected branch) -> identify bad-mu / bad-var / both
                    paths.append(plot_null_comparison(
                        rep[f"fit_r2_branch_{tag}_mu"], rep[f"fit_r2_branch_{tag}_var"],
                        os.path.join(out_dir, f"branch_fit_r2_{tag}_muvar_scatter.png"),
                        xlabel=f"{tag} R2 (mu)", ylabel=f"{tag} R2 (var)", clip_min=-1.0,
                        title=f"per-branch {tag} R2  mu vs var (all cells)"))
                    paths.append(plot_null_comparison(
                        rep[f"fit_r2_branch_{tag}_mu_conf"], rep[f"fit_r2_branch_{tag}_var_conf"],
                        os.path.join(out_dir, f"branch_fit_r2_{tag}_muvar_scatter_conf.png"),
                        xlabel=f"{tag} R2 (mu)", ylabel=f"{tag} R2 (var)", clip_min=-1.0,
                        title=f"per-branch {tag} R2  mu vs var (confident)"))
                # mean-null vs line-null (combined)
                paths.append(plot_null_comparison(
                    rep["fit_r2_branch_meanNull"], rep["fit_r2_branch_lineNull"],
                    os.path.join(out_dir, "meanNull_vs_lineNull_scatter.png"), clip_min=-1.0))
                for p in paths:
                    print(f"[{_now()}] wrote {p}")
            except Exception as e:
                print(f"  [warn] scatter plots skipped: {e}")

        df = pd.DataFrame(csv_cols)
        out_csv = out_csv or os.path.join(out_dir, "gene_fit_scores_v2.csv")
        df.to_csv(out_csv, index=False)
        print(f"[{_now()}] wrote {out_csv}")
        if write_h5ad:
            out_h5ad = out_h5ad or os.path.join(out_dir, os.path.basename(path))
            adata.write_h5ad(out_h5ad)
            print(f"[{_now()}] wrote {out_h5ad}")
        print(f"[{_now()}] done in {time.time()-t0:.1f}s")
    finally:
        sys.stdout = old; tee.close()
        print(f"log written to {os.path.abspath(log_file)}")


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(0)
    N, G, S = 1500, 5, 4
    mu = np.abs(rng.normal(35, 18, (N, G))); branch = rng.integers(0, 2, (N, G))
    var = np.where(branch == 0, 60 * mu - 0.35 * mu ** 2, 0.3 * mu ** 2) + rng.normal(0, 25, (N, G))
    mf = mu.copy(); vf = np.where(branch == 0, 60 * mu - 0.35 * mu ** 2, 0.3 * mu ** 2)
    prob = np.zeros((N, G, S))
    prob[..., 0] = np.where(branch == 0, 0.85, 0.06); prob[..., 1] = np.where(branch == 0, 0.07, 0.04)
    prob[..., 2] = np.where(branch == 1, 0.85, 0.06); prob[..., 3] = np.where(branch == 1, 0.07, 0.04)
    prob /= prob.sum(-1, keepdims=True)
    rep = confidence_report(mu, var, mf, vf, prob, conf_thresh=0.9, branch_min_cells=30)
    print_confidence_report(rep, 0.9, 20)
    ok = (np.nanmedian(rep["fit_r2_branch_meanNull"]) > 0.5 and
          np.nanmedian(rep["fit_r2_branch_lineNull"]) > 0.3)
    print(f"\n[selftest] both nulls positive & sensible -> {'PASS' if ok else 'FAIL'}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5ad", nargs="?")
    p.add_argument("--mu_obs_layer", default="mu_scvi_smooth")
    p.add_argument("--var_obs_layer", default="var_scvi_smooth")
    p.add_argument("--mu_fit_layer", default="mu_fit")
    p.add_argument("--var_fit_layer", default="var_fit")
    p.add_argument("--combine", default="min", choices=["min", "mean"])
    p.add_argument("--prob_state", default=None)
    p.add_argument("--conf_thresh", type=float, default=0.9)
    p.add_argument("--target_error", type=float, default=None)
    p.add_argument("--up_states", type=int, nargs="+", default=[0, 1])
    p.add_argument("--down_states", type=int, nargs="+", default=[2, 3])
    p.add_argument("--branch_min_cells", type=int, default=30)
    p.add_argument("--branch_agg", default="max", choices=["max", "min", "weighted"])
    p.add_argument("--max_drop_pct", type=float, default=20.0)
    p.add_argument("--thresh_meannull", type=float, default=0.5,
                   help="reliability threshold on fit_r2_branch_meanNull")
    p.add_argument("--thresh_linenull", type=float, default=0.0,
                   help="reliability threshold on fit_r2_branch_lineNull")
    p.add_argument("--write_h5ad", action="store_true")
    p.add_argument("--out_h5ad", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--log_file", default=None)
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    if a.selftest:
        selftest()
    elif a.h5ad:
        run_on_h5ad(a.h5ad, mu_obs_layer=a.mu_obs_layer, var_obs_layer=a.var_obs_layer,
                    mu_fit_layer=a.mu_fit_layer, var_fit_layer=a.var_fit_layer,
                    combine=a.combine, write_h5ad=a.write_h5ad, out_h5ad=a.out_h5ad,
                    out_csv=a.out_csv, log_file=a.log_file, out_dir=a.out_dir,
                    diagnose_only=a.diagnose, prob_state=a.prob_state,
                    conf_thresh=a.conf_thresh, up_states=tuple(a.up_states),
                    down_states=tuple(a.down_states), max_drop_pct=a.max_drop_pct,
                    target_error=a.target_error, branch_min_cells=a.branch_min_cells,
                    branch_agg=a.branch_agg, thresh_meannull=a.thresh_meannull,
                    thresh_linenull=a.thresh_linenull)
    else:
        raise SystemExit("provide an h5ad path or --selftest")
