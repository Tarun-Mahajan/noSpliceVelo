"""Plotting and linear-model pipeline for Cross-Boundary Direction (CBDir) results.

Companion to `compute_cbdir_run.py`, mirroring `plot_velocity_confidence.py`.

Features:
- Discovers dataset paths and method names from the CBDir global + per-method YAML configs.
- Loads each dataset's `{dataset}_cbdir_long{suffix}.csv` produced by compute_cbdir_run.py.
- Aggregated boxplot: x = dataset, y = CBDir, hue = method (no individual points).
  Emitted twice: in the user's specified method order, and with methods ranked by
  decreasing median CBDir within each dataset ("_ranked" filenames).
- Per-dataset boxplots: x = transition edge, hue = method, edges ordered as in cluster_edges.
- Paired Wilcoxon significance stars against a reference method, paired on
  obs_id = f"{cell_barcode}___{edge}" (NOT on cell_barcode, which recurs across edges).
- Per-method offset vs a reference, from paired per-cell differences on the
  Fisher-z (arctanh) scale, reported on both scales. The unit of replication is
  configurable: `edge` (default) averages within each transition and tests across
  transitions, which respects the fact that cells inside a transition are
  correlated; `cell` treats every cell as independent and is anticonservative.
  Both are computed in closed form, so there is no optimizer to fail to converge.
- Forest plot of offsets with 95% CIs, and residual diagnostics.

Usage:
    python plot_cbdir.py --global-config cbdir_global_config_4throot.yaml \
                         --plot-config plot_cbdir_config_nsv_comparison.yaml [--str-suffix _v2]
"""

import os
import re
import itertools
import sys
import time
import argparse
import warnings
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (summary stability figure)
import matplotlib.transforms  # noqa: F401
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# What the plotted quantity and the within-dataset replication unit are called
# on axis labels and titles. The defaults are this pipeline's own vocabulary, so
# nothing here changes a CBDir run. They are indirected only so that another
# pipeline whose data has the same shape can reuse this whole figure suite under
# its own names: plot_velocity_confidence.py drives it with
# VALUE_LABEL="Velocity confidence" and UNIT_LABEL="cluster", where the long
# table's `edge` column carries cell-type clusters rather than transitions.
#
# Read at call time, never captured at import, so `with vocabulary(...)` around
# a call is enough.

VALUE_LABEL = "CBDir"
UNIT_LABEL = "transition"
UNIT_LABEL_PLURAL = "transitions"


@contextmanager
def vocabulary(value_label=None, unit_label=None, unit_label_plural=None):
    """Temporarily rename the plotted value and the replication unit."""
    global VALUE_LABEL, UNIT_LABEL, UNIT_LABEL_PLURAL
    prev = (VALUE_LABEL, UNIT_LABEL, UNIT_LABEL_PLURAL)
    try:
        if value_label:
            VALUE_LABEL = value_label
        if unit_label:
            UNIT_LABEL = unit_label
            UNIT_LABEL_PLURAL = unit_label_plural or (unit_label + "s")
        elif unit_label_plural:
            UNIT_LABEL_PLURAL = unit_label_plural
        yield
    finally:
        VALUE_LABEL, UNIT_LABEL, UNIT_LABEL_PLURAL = prev


# ---------------------------------------------------------------------------
# Small shared helpers (mirroring plot_velocity_confidence.py)
# ---------------------------------------------------------------------------

def draw_zero_line(ax, plot_cfg, vertical=True):
    """Mark CBDir = 0 — the boundary between correct and reversed direction.

    Drawn ABOVE the grid, because seaborn's whitegrid puts a tick line at 0.0 on
    most of these axes and a zorder-0 rule is invisible underneath it. It still
    sits below the boxes and points (zorder 2+), so it shows in the gaps rather
    than striking through the data.

    `vertical=True` means CBDir is on the y axis, so the rule is horizontal.
    """
    if not plot_cfg.get("zero_line", True):
        return None
    draw = ax.axhline if vertical else ax.axvline
    return draw(0.0,
                color=plot_cfg.get("zero_line_color", "#333333"),
                linestyle=plot_cfg.get("zero_line_style", "--"),
                linewidth=_cfg_num(plot_cfg, "zero_line_width", 1.2),
                zorder=_cfg_num(plot_cfg, "zero_line_zorder", 1.6),
                dashes=(5, 3))


def _cfg_num(plot_cfg, key, default, cast=float):
    """A numeric config value, treating an explicit `null` as "not set".

    YAML's null arrives as None, and `int(None)` raises. Several options are
    documented as `null` to mean "derive it", so every numeric read goes through
    here rather than through int()/float() on a raw .get().
    """
    v = plot_cfg.get(key, None)
    if v is None or (isinstance(v, str) and not v.strip()):
        return cast(default)
    try:
        return cast(v)
    except (TypeError, ValueError):
        print(f"  WARNING: config '{key}' = {v!r} is not a number; using {default!r}")
        return cast(default)


def _cfg_int(plot_cfg, key, default):
    return _cfg_num(plot_cfg, key, default, cast=int)


def pvalue_to_stars(p):
    """Convert p-value to significance stars notation."""
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def resolve_ylim(values, plot_cfg, annot_frac=0.0):
    """Axis limits, plus where annotations should start.

    ylim_from_data (default true) fits the axis to the plotted values with 5%
    padding, so a panel whose CBDir never leaves [-0.4, 0.6] is not squashed into
    a fixed [-1, 1] frame. Set it false to pin every figure to `ylim`, which is
    what you want when panels must be compared side by side across runs.

    Returns (axis_low, axis_high, annotation_base). annot_frac reserves that
    fraction of the data span above the data for the annotation block.
    """
    v = np.asarray([x for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if bool(plot_cfg.get("ylim_from_data", True)) and v.size:
        lo, hi = float(np.min(v)), float(np.max(v))
        span = (hi - lo) or 1.0
        pad = 0.05 * span
        lo_ax, annot_base = lo - pad, hi + pad
    else:
        cfg = plot_cfg.get("ylim", [-1.0, 1.0])
        lo_ax, annot_base = float(cfg[0]) - 0.05, float(cfg[1])
        span = (float(cfg[1]) - float(cfg[0])) or 1.0
    return lo_ax, annot_base + annot_frac * span, annot_base


def fmt_p(p, prefix="p="):
    """Compact, readable p-value: 3 significant figures, scientific when small."""
    if p is None or not np.isfinite(p):
        return f"{prefix}NA"
    if p < 1e-3:
        mant, exp = f"{p:.0e}".split("e")
        return f"{prefix}{mant}e{int(exp)}"
    return f"{prefix}{p:.3g}"


# frame -> (adjusted p column, df column, short label for the figure)
_FRAME_COLS = {
    "dataset":       ("p_dataset_adj", "df_dataset", "ds-rand"),
    "dataset_fixed": ("p_dsfixed_adj", "df_dsfixed", "ds-fixed"),
    "edge":          ("p_edge_adj",    "df_edge",    "edge"),
}


def _save_figure(out_base, dpi):
    """Save the current figure as PNG + PDF, creating the directory if needed."""
    save_dir = os.path.dirname(os.path.abspath(out_base))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    png_path, pdf_path = f"{out_base}.png", f"{out_base}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def _clean_save_base(save_path):
    if save_path.endswith(".png") or save_path.endswith(".pdf"):
        return os.path.splitext(save_path)[0]
    return save_path


def _suffix_str(str_suffix):
    if str_suffix is None or str(str_suffix).strip() == "":
        return ""
    s = str(str_suffix).strip()
    return s if (s.startswith("_") or s.startswith("-")) else f"_{s}"


# ---------------------------------------------------------------------------
# Run log (same tee/filter machinery as compute_cbdir_run.py)
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for st in self._streams:
            try:
                st.write(data)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def close(self):
        self.flush()


class _FilteredFile:
    """Write-through file that drops progress-bar / spinner lines."""

    def __init__(self, fh):
        self._fh = fh
        self._buf = ""

    def write(self, data):
        import re
        self._buf += data
        _PROGRESS_RE = re.compile(r"^\s*(loss|epoch|\d+%|\[|\.|\*|-)", re.IGNORECASE)
        while "\n" in self._buf:
            nl = self._buf.find("\n")
            cr = self._buf.rfind("\r", 0, nl)
            line = self._buf[cr + 1:nl] if cr != -1 else self._buf[:nl]
            self._buf = self._buf[nl + 1:]
            eff = line.rsplit("\r", 1)[-1]
            if eff.strip() and _PROGRESS_RE.search(eff):
                continue
            self._fh.write(eff + "\n")
        self._fh.flush()

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self):
        import re
        rem = self._buf.rsplit("\r", 1)[-1]
        _PROGRESS_RE = re.compile(r"^\s*(loss|epoch|\d+%|\[|\.|\*|-)", re.IGNORECASE)
        if rem.strip() and not _PROGRESS_RE.search(rem):
            self._fh.write(rem)
        self._buf = ""
        self._fh.flush()
        self._fh.close()


def _resolve_log_file(log_file):
    """Turn the configured value into a path, or None for stdout only.

    True / "" / "auto"  -> timestamped plot_cbdir_YYYYmmdd_HHMMSS.log in the cwd
    a path              -> that path (parent directories are created)
    None / false        -> no log file
    """
    if log_file is None or log_file is False:
        return None
    if log_file is True or str(log_file).strip().lower() in ("", "auto", "true"):
        return f"plot_cbdir_{time.strftime('%Y%m%d_%H%M%S')}.log"
    return str(log_file)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _forward_transform(v, kind="atanh", eps=1e-6):
    """Map CBDir onto the modelling scale. Returns (values, n_clipped)."""
    v = np.asarray(v, dtype=np.float64)
    if kind in (None, "none"):
        return v, 0
    if kind == "atanh":
        lo, hi = -1.0 + eps, 1.0 - eps
        clipped = np.clip(v, lo, hi)
        n_clipped = int(np.sum(~np.isclose(clipped, v, rtol=0, atol=0)))
        return np.arctanh(clipped), n_clipped
    if kind == "logit":
        p = np.clip((v + 1.0) / 2.0, eps, 1.0 - eps)
        n_clipped = int(np.sum(~np.isclose(p, (v + 1.0) / 2.0, rtol=0, atol=0)))
        return np.log(p / (1.0 - p)), n_clipped
    raise ValueError(f"transform must be none | atanh | logit, got '{kind}'")


def _inverse_transform(z, kind="atanh"):
    """Map the modelling scale back to CBDir units."""
    z = np.asarray(z, dtype=np.float64)
    if kind in (None, "none"):
        return z
    if kind == "atanh":
        return np.tanh(z)
    if kind == "logit":
        return 2.0 / (1.0 + np.exp(-z)) - 1.0
    raise ValueError(f"transform must be none | atanh | logit, got '{kind}'")


def _native_offset(z_ref_mean, offset_z, kind="atanh"):
    """Offset in CBDir units, evaluated at the reference's own level.

    A difference of transformed values does NOT back-transform elementwise:
    tanh(offset_z) is not the offset in CBDir units. The offset is the change in
    CBDir when moving the reference's mean transformed value by offset_z.
    """
    if kind in (None, "none"):
        return float(offset_z)
    hi = _inverse_transform(z_ref_mean + offset_z, kind)
    lo = _inverse_transform(z_ref_mean, kind)
    return float(hi - lo)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cbdir_long(global_config_path, str_suffix=None):
    """Discover methods/datasets from the CBDir configs and load the long tables.

    Returns (long_df, discovered_methods, dataset_order, dataset_dir_paths,
             edges_by_dataset).
    """
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML is required. Install it with `pip install pyyaml`.") from e

    with open(global_config_path, "r") as fh:
        global_cfg = yaml.safe_load(fh) or {}

    if "methods" not in global_cfg:
        raise ValueError("Global config must contain a 'methods:' mapping.")

    if str_suffix is None:
        str_suffix = global_cfg.get("str_suffix", None)
    suffix = _suffix_str(str_suffix)

    methods_dict = global_cfg["methods"]
    global_datasets = global_cfg.get("datasets", {}) or {}

    discovered_methods = []
    dataset_dir_paths = {}
    dataset_order = []

    for method_name, method_cfg_path in methods_dict.items():
        if not os.path.exists(method_cfg_path):
            print(f"WARNING: method config not found for '{method_name}': {method_cfg_path}")
            continue
        with open(method_cfg_path, "r") as fh:
            m_cfg = yaml.safe_load(fh) or {}
        if "datasets" not in m_cfg:
            continue
        discovered_methods.append(method_name)
        for entry in m_cfg.get("datasets", []):
            d_name = entry.get("name")
            if not d_name or "dir_path" not in entry:
                continue
            dataset_dir_paths.setdefault(d_name, entry["dir_path"])
            if d_name not in dataset_order:
                dataset_order.append(d_name)

    # Edge order per dataset, straight from cluster_edges (not alphabetical).
    edges_by_dataset = {}
    for d_name in dataset_order:
        raw_edges = (global_datasets.get(d_name, {}) or {}).get("cluster_edges")
        labels = []
        for e in (raw_edges or []):
            if isinstance(e, str) and "->" in e:
                u, v = e.split("->", 1)
                labels.append(f"{u.strip()} -> {v.strip()}")
            elif isinstance(e, (list, tuple)) and len(e) == 2:
                labels.append(f"{e[0]} -> {e[1]}")
        edges_by_dataset[d_name] = labels

    frames = []
    for d_name in dataset_order:
        dir_path = dataset_dir_paths[d_name]
        path = os.path.join(dir_path, f"{d_name}_cbdir_long{suffix}.csv")
        if not os.path.exists(path):
            print(f"WARNING: long table not found for dataset '{d_name}': {path}")
            continue
        df = pd.read_csv(path)
        frames.append(df)
        print(f"  Loaded {path}  ({len(df)} rows)")

    if not frames:
        raise FileNotFoundError(
            "No CBDir long tables found. Run compute_cbdir_run.py first, and check "
            "that --str-suffix matches the suffix used there."
        )

    long_df = pd.concat(frames, ignore_index=True)

    n_before = len(long_df)
    long_df = long_df[long_df["cbdir"].notna()].copy()
    if len(long_df) < n_before:
        print(f"  Dropped {n_before - len(long_df)} rows with NaN cbdir "
              f"(from include_skipped=true runs).")

    # The unit that is measured once per method. A barcode recurs across edges,
    # so pairing on cell_barcode alone would mis-align methods. The dataset is in
    # the key too: edge labels are not globally unique (two datasets can both have
    # "HSC_1 -> HSC_2"), and any analysis that pivots across datasets — the
    # logistic tests do — would otherwise silently merge cells from different
    # datasets that happen to share a barcode and an edge name.
    long_df["obs_id"] = (long_df["dataset"].astype(str) + "___"
                         + long_df["cell_barcode"].astype(str) + "___"
                         + long_df["edge"].astype(str))

    dataset_order = [d for d in dataset_order if d in set(long_df["dataset"])]
    discovered_methods = [m for m in discovered_methods if m in set(long_df["method"])]

    for d_name in dataset_order:
        seen = list(pd.unique(long_df.loc[long_df["dataset"] == d_name, "edge"]))
        declared = [e for e in edges_by_dataset.get(d_name, []) if e in seen]
        edges_by_dataset[d_name] = declared + [e for e in seen if e not in declared]

    return long_df, discovered_methods, dataset_order, dataset_dir_paths, edges_by_dataset


# ---------------------------------------------------------------------------
# Method ordering / palette
# ---------------------------------------------------------------------------

def compute_grouped_method_order(long_df, method_groups_cfg, method_order_cfg, discovered_methods):
    """Order methods by group, then by mean-across-datasets of within-dataset median."""
    def _rank(methods):
        scores = {}
        for m in methods:
            sub = long_df[long_df["method"] == m]
            if sub.empty:
                scores[m] = -np.inf
                continue
            medians = sub.groupby("dataset")["cbdir"].median()
            scores[m] = float(np.mean(medians)) if len(medians) else -np.inf
        return sorted(methods, key=lambda m: scores[m], reverse=True)

    if method_groups_cfg and isinstance(method_groups_cfg, dict):
        ordered = []
        for gname in sorted(method_groups_cfg.keys()):
            members = [m for m in (method_groups_cfg[gname] or []) if m in discovered_methods]
            ordered.extend(_rank(members))
        ordered.extend([m for m in discovered_methods if m not in ordered])
        return ordered

    if method_order_cfg:
        ordered = [m for m in method_order_cfg if m in discovered_methods]
        ordered.extend([m for m in discovered_methods if m not in ordered])
        return ordered

    return _rank(list(discovered_methods))


def build_palette(method_order, method_colors_cfg):
    default_palette = sns.color_palette("Set2", n_colors=max(len(method_order), 1))
    palette = {}
    for idx, m in enumerate(method_order):
        if method_colors_cfg and isinstance(method_colors_cfg, dict) and m in method_colors_cfg:
            palette[m] = method_colors_cfg[m]
        else:
            palette[m] = default_palette[idx]
    return palette


# ---------------------------------------------------------------------------
# Significance stars (shared by both plot types)
# ---------------------------------------------------------------------------

def _paired_pvalue(ref_series, target_series, stat_test):
    """Paired test on obs_id-aligned values. Returns (p, n_pairs)."""
    aligned = pd.concat([ref_series, target_series], axis=1, keys=["ref", "target"]).dropna()
    if len(aligned) < 3:
        return None, len(aligned)
    try:
        if stat_test == "wilcoxon":
            diff = aligned["target"].to_numpy() - aligned["ref"].to_numpy()
            if np.allclose(diff, 0):
                return 1.0, len(aligned)
            _, p = stats.wilcoxon(aligned["target"], aligned["ref"])
        else:
            _, p = stats.mannwhitneyu(aligned["target"], aligned["ref"], alternative="two-sided")
        return float(p), len(aligned)
    except Exception as e:
        print(f"    WARNING: stat test failed: {e}")
        return None, len(aligned)


def _methods_by_x(data, x_col, x_order, method_order, per_x_order):
    """Method draw-order within each x category.

    With per_x_order=True the methods are ranked by decreasing median CBDir
    *within that x category*, so the best-performing method sits leftmost in
    every dataset. The full method list is always returned (never filtered), so
    slot count — and therefore box width — stays identical across categories;
    a method with no cells there simply draws nothing. Methods absent from a
    category sort to the end.
    """
    out = {}
    for xv in x_order:
        sub = data[data[x_col] == xv]
        if per_x_order:
            med = sub.groupby("method")["cbdir"].median()
            out[xv] = sorted(method_order,
                             key=lambda m: med.get(m, -np.inf)
                             if np.isfinite(med.get(m, -np.inf)) else -np.inf,
                             reverse=True)
        else:
            out[xv] = list(method_order)
    return out


def _draw_grouped_boxplot(ax, data, x_col, x_order, methods_by_x, palette,
                          box_linewidth=0.7, showfliers=False, width=0.8):
    """Grouped boxplot drawn directly on matplotlib.

    seaborn's `hue_order` is global across x categories, so it cannot express a
    per-category method order. Drawing the boxes here keeps the two orderings on
    one code path and gives exact box positions for the star annotations.
    Returns {(x_value, method): x_position}.
    """
    n_slots = max((len(v) for v in methods_by_x.values()), default=1)
    box_width = width / max(n_slots, 1)
    pos_lookup = {}

    for x_idx, x_val in enumerate(x_order):
        methods_here = methods_by_x.get(x_val, [])
        for m_idx, m_name in enumerate(methods_here):
            pos = x_idx - (width / 2) + (m_idx + 0.5) * box_width
            pos_lookup[(x_val, m_name)] = pos
            vals = data.loc[(data[x_col] == x_val) & (data["method"] == m_name),
                            "cbdir"].dropna().to_numpy()
            if vals.size == 0:
                continue
            bp = ax.boxplot(
                [vals], positions=[pos], widths=box_width * 0.88,
                patch_artist=True, showfliers=showfliers, manage_ticks=False,
                boxprops=dict(linewidth=box_linewidth, edgecolor="#333333"),
                whiskerprops=dict(linewidth=box_linewidth, color="#333333"),
                capprops=dict(linewidth=box_linewidth, color="#333333"),
                medianprops=dict(linewidth=box_linewidth * 1.6, color="#111111"),
                flierprops=dict(marker=".", markersize=2, markeredgewidth=0.3,
                                markerfacecolor="#666666", markeredgecolor="#666666"),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(palette.get(m_name, "#888888"))
                patch.set_alpha(0.85)

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(list(x_order))
    ax.set_xlim(-0.6, len(x_order) - 0.4)
    return pos_lookup


def _method_legend(ax, method_order, palette, title="Method"):
    """Legend built from proxy patches (the boxes are raw matplotlib artists)."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette.get(m, "#888888"), edgecolor="#333333",
                     alpha=0.85, label=m) for m in method_order]
    ax.legend(handles=handles, title=title, bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=True)


def _annotate_stars(ax, data, x_col, x_order, pos_lookup, reference_method,
                    stat_test, star_rotation, y_cap, pvalues=None):
    """Annotate significance stars above each non-reference box.

    `pvalues` maps (x_value, method) -> p. When given (the model p-values), it is
    used directly; otherwise a paired Wilcoxon is computed on the spot. The model
    p-values are preferred because the Wilcoxon treats every cell as an
    independent replicate, which cells within a transition are not.
    """
    for x_val in x_order:
        sub = data[data[x_col] == x_val]
        ref_series = sub[sub["method"] == reference_method].set_index("obs_id")["cbdir"]
        if ref_series.empty:
            continue
        for (xv, m_name), x_pos in pos_lookup.items():
            if xv != x_val or m_name == reference_method:
                continue
            target_series = sub[sub["method"] == m_name].set_index("obs_id")["cbdir"]
            if target_series.empty:
                continue
            if pvalues is not None:
                if (x_val, m_name) not in pvalues:
                    continue
                p_val = pvalues[(x_val, m_name)]
            else:
                p_val, _ = _paired_pvalue(ref_series, target_series, stat_test)
            if p_val is None or not np.isfinite(p_val):
                continue
            stars = pvalue_to_stars(p_val)
            vals = target_series.to_numpy()
            q75, q25 = np.percentile(vals, 75), np.percentile(vals, 25)
            upper_whisker = min(np.max(vals), q75 + 1.5 * (q75 - q25))
            y_text = min(upper_whisker + 0.03, y_cap)
            ax.text(x_pos, y_text, stars, ha="center", va="bottom",
                    rotation=star_rotation, fontsize=9,
                    fontweight="bold" if stars != "ns" else "normal",
                    color="#333333" if stars == "ns" else "#d95f02")


# ---------------------------------------------------------------------------
# Plot 1 — aggregated across datasets
# ---------------------------------------------------------------------------

def _resolve_variants(cfg_value, default):
    """Normalise an order-variant list. 'specified' | 'ranked'."""
    if cfg_value is None:
        cfg_value = default
    if isinstance(cfg_value, str):
        cfg_value = [cfg_value]
    out = []
    for v in cfg_value:
        v = str(v).strip().lower()
        if v in ("specified", "user", "global"):
            out.append("specified")
        elif v in ("ranked", "median", "sorted"):
            out.append("ranked")
        else:
            print(f"WARNING: unknown order variant '{v}' (use specified | ranked); ignored.")
    return out or ["specified"]


def _resolve_point_units(cfg_value):
    """Which replication unit the pooled figure draws: transition | dataset."""
    if cfg_value is None:
        cfg_value = ["transition", "dataset"]
    if isinstance(cfg_value, str):
        cfg_value = [cfg_value]
    out = []
    for v in cfg_value:
        v = str(v).strip().lower()
        if v in ("transition", "edge", "transitions"):
            out.append("transition")
        elif v in ("dataset", "datasets"):
            out.append("dataset")
        else:
            print(f"WARNING: unknown pooled point unit '{v}' "
                  f"(use transition | dataset); ignored.")
    return out or ["transition"]


def _variant_suffix(variant):
    return "_ranked" if variant == "ranked" else ""


def plot_aggregated(long_df, method_order, dataset_order, plot_cfg, suffix,
                    pvalues=None, pwin=None):
    """Aggregated boxplot: x = dataset, hue = method.

    Emits one figure per order variant:
      specified -> the user's method_groups / method_order, shared by all datasets
      ranked    -> methods ordered by decreasing median CBDir within each dataset
    """
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    figsize = tuple(plot_cfg.get("figsize", [14, 6]))
    dpi = plot_cfg.get("dpi", 300)
    ylim = plot_cfg.get("ylim", [-1.0, 1.0])
    aggregate_by = str(plot_cfg.get("aggregate_by", "cells")).lower()
    reference_method = plot_cfg.get("reference_method")

    data = long_df[long_df["method"].isin(method_order)].copy()
    if aggregate_by == "edges":
        data = data.groupby(["dataset", "method", "edge"], as_index=False)["cbdir"].mean()
        data["obs_id"] = data["edge"]
        ylabel_extra = " (per-edge means)"
    else:
        ylabel_extra = ""

    variants = _resolve_variants(plot_cfg.get("aggregated_order_variants"),
                                 ["specified", "ranked"])
    annots = plot_cfg.get("aggregated_annotation_variants", ["none", "pwin"])
    if isinstance(annots, str):
        annots = [annots]
    annots = [str(a).strip().lower() for a in annots] or ["none"]

    lo_ax, hi_ax, annot_base = resolve_ylim(data["cbdir"], plot_cfg, annot_frac=0.22)
    span = max(annot_base - lo_ax, 1e-9)

    for variant in variants:
        per_x = (variant == "ranked")
        methods_by_x = _methods_by_x(data, "dataset", dataset_order, method_order, per_x)
        print(f"  [{variant}] method order per dataset:")
        for d in dataset_order:
            print(f"    {d}: {methods_by_x[d]}")

        for annot in annots:
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            pos_lookup = _draw_grouped_boxplot(
                ax, data, "dataset", dataset_order, methods_by_x, palette,
                box_linewidth=plot_cfg.get("box_linewidth", 0.7),
                showfliers=plot_cfg.get("show_points", False))

            draw_zero_line(ax, plot_cfg)

            # Significance stars are deliberately NOT drawn here: a per-dataset
            # test on a handful of transitions is too underpowered to label, and
            # a wall of "ns" reads as "no difference" when it means "not provable
            # from this many transitions". Inference lives on the pooled figure.
            if annot == "pwin" and pwin:
                for (x_val, m_name), x_pos in pos_lookup.items():
                    if (x_val, m_name) not in pwin:
                        continue
                    sub = data[(data["dataset"] == x_val) & (data["method"] == m_name)]
                    if sub.empty:
                        continue
                    vals = sub["cbdir"].to_numpy()
                    q75, q25 = np.percentile(vals, 75), np.percentile(vals, 25)
                    yv = min(min(np.max(vals), q75 + 1.5 * (q75 - q25)) + 0.02 * span,
                             annot_base)
                    ax.text(x_pos, yv, f"{pwin[(x_val, m_name)]:.2f}".lstrip("0"),
                            ha="center", va="bottom",
                            rotation=plot_cfg.get("star_rotation", 90),
                            fontsize=7, color="#444444")

            title = plot_cfg.get("title",
                                 "Cross-Boundary Direction Correctness Across Datasets")
            if per_x:
                title += " (methods ranked within each dataset)"
            if annot == "pwin":
                title += f"\nannotated with P(method > {plot_cfg.get('reference_method')})"
            ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
            ax.set_xlabel(plot_cfg.get("xlabel", "Dataset"), fontsize=13, fontweight="bold")
            ax.set_ylabel(plot_cfg.get("ylabel", VALUE_LABEL) + ylabel_extra,
                          fontsize=13, fontweight="bold")
            ax.set_ylim(lo_ax, hi_ax)
            rot = plot_cfg.get("xlabel_rotation", 45)
            plt.setp(ax.get_xticklabels(), rotation=rot,
                     ha="right" if rot != 0 else "center")
            _method_legend(ax, method_order, palette)
            fig.tight_layout()
            a_sfx = "_pwin" if annot == "pwin" else ""
            _save_figure(
                f"{save_base}_aggregated{_variant_suffix(variant)}{a_sfx}{suffix}", dpi)




# ---------------------------------------------------------------------------
# Plot 2 - per dataset, x = edge
# ---------------------------------------------------------------------------

def plot_per_dataset(long_df, method_order, dataset_order, edges_by_dataset, plot_cfg, suffix):
    """One figure per dataset: x = transition edge, hue = method.

    Emits one figure per order variant, as for the aggregated plot; here
    'ranked' orders methods by median CBDir within each individual edge.
    """
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    figsize = tuple(plot_cfg.get("per_dataset_figsize", [16, 6]))
    dpi = plot_cfg.get("dpi", 300)
    ylim = plot_cfg.get("ylim", [-1.0, 1.0])
    reference_method = plot_cfg.get("reference_method")

    variants = _resolve_variants(plot_cfg.get("per_dataset_order_variants"),
                                 ["specified", "ranked"])

    for d_name in dataset_order:
        data = long_df[(long_df["dataset"] == d_name) & (long_df["method"].isin(method_order))]
        if data.empty:
            continue
        edge_order = edges_by_dataset.get(d_name) or sorted(pd.unique(data["edge"]))
        lo_ax, hi_ax, _ = resolve_ylim(data["cbdir"], plot_cfg, annot_frac=0.06)

        for variant in variants:
            per_x = (variant == "ranked")
            methods_by_x = _methods_by_x(data, "edge", edge_order, method_order, per_x)

            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            pos_lookup = _draw_grouped_boxplot(
                ax, data, "edge", edge_order, methods_by_x, palette,
                box_linewidth=plot_cfg.get("box_linewidth", 0.7),
                showfliers=plot_cfg.get("show_points", False))

            draw_zero_line(ax, plot_cfg)

            # No stars here either: within a single transition there is no
            # edge-clustered test to draw on, and the cell-level one is
            # anticonservative. These panels are descriptive.

            title = f"{plot_cfg.get('per_dataset_title', f'{VALUE_LABEL} by {UNIT_LABEL}')} - {d_name}"
            if per_x:
                title += f" (methods ranked within each {UNIT_LABEL})"
            ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
            ax.set_xlabel(plot_cfg.get("edge_xlabel", UNIT_LABEL.capitalize()),
                          fontsize=13, fontweight="bold")
            ax.set_ylabel(plot_cfg.get("ylabel", VALUE_LABEL), fontsize=13, fontweight="bold")
            ax.set_ylim(lo_ax, hi_ax)
            rot = plot_cfg.get("xlabel_rotation", 45)
            plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot != 0 else "center")
            _method_legend(ax, method_order, palette)
            fig.tight_layout()
            _save_figure(f"{save_base}_{d_name}_by_edge{_variant_suffix(variant)}{suffix}", dpi)


# ---------------------------------------------------------------------------
# Per-method panels: one method against the reference, every transition at once
# ---------------------------------------------------------------------------

def _slug(name):
    """Filesystem-safe stem for a method name."""
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_")
    return s or "method"


def per_method_edge_tests(long_df, method_order, plot_cfg):
    """Per-edge paired Wilcoxon of each method against the reference.

    One test per (dataset, transition, method), on the cells the method and the
    reference both scored. Correction is applied **within each dataset, across
    that dataset's transitions** — the family is "the transitions of this dataset",
    which is what makes the per-dataset panel readable as a unit.

    These are cell-level tests, so they are pseudoreplicated with respect to any
    claim about the method overall: read a single row as "on this transition the
    two methods' scores separate", never as evidence that the method is better.
    The transition- and dataset-level tests elsewhere in this module are what
    carry that claim.
    """
    ref = plot_cfg.get("reference_method")
    if not ref or ref not in method_order:
        return pd.DataFrame()
    stat_test = str(plot_cfg.get("stat_test", "wilcoxon")).lower()
    p_adjust = str(plot_cfg.get("per_method_p_adjust")
                   or plot_cfg.get("p_adjust", "fdr_bh")).lower()
    from statsmodels.stats.multitest import multipletests

    rows = []
    for m in method_order:
        if m == ref:
            continue
        sub = long_df[long_df["method"].isin([ref, m])]
        if sub.empty:
            continue
        for d_name, dsub in sub.groupby("dataset", sort=False):
            recs = []
            for e_name, esub in dsub.groupby("edge", sort=False):
                w = esub.pivot_table(index="obs_id", columns="method",
                                     values="cbdir", aggfunc="first")
                if ref not in w.columns or m not in w.columns:
                    continue
                p, n = _paired_pvalue(w[ref], w[m], stat_test)
                med_m = esub.loc[esub["method"] == m, "cbdir"].median()
                med_r = esub.loc[esub["method"] == ref, "cbdir"].median()
                recs.append(dict(method=m, reference_method=ref, dataset=d_name,
                                 edge=e_name, n_pairs=int(n),
                                 median_method=float(med_m) if pd.notna(med_m) else np.nan,
                                 median_reference=float(med_r) if pd.notna(med_r) else np.nan,
                                 p=np.nan if p is None else float(p)))
            if not recs:
                continue
            v = np.array([r["p"] for r in recs], dtype=float)
            ok = np.where(np.isfinite(v))[0]
            adj = np.full(v.size, np.nan)
            if ok.size:
                adj[ok] = (multipletests(v[ok], method=p_adjust)[1]
                           if p_adjust != "none" else v[ok])
            for r, a in zip(recs, adj):
                r["p_adj"] = float(a) if np.isfinite(a) else np.nan
                r["n_edges_in_dataset"] = int(len(recs))
                r["p_adjust"] = p_adjust
            rows.extend(recs)
    return pd.DataFrame(rows)


def _per_method_row_order(data, ref, m, dataset_order, edges_by_dataset, how):
    """(dataset, edge) rows, datasets then edges by decreasing median CBDir.

    The median is taken over the cells of BOTH plotted methods, so the ordering is
    a property of the transition rather than of whichever method is on the page —
    otherwise every method would reshuffle the axis and the panels could not be
    compared side by side.
    """
    if how != "median":
        rows = []
        for d in dataset_order:
            for e in (edges_by_dataset.get(d)
                      or sorted(pd.unique(data.loc[data["dataset"] == d, "edge"]))):
                if ((data["dataset"] == d) & (data["edge"] == e)).any():
                    rows.append((d, e))
        return rows
    ds_med = data.groupby("dataset")["cbdir"].median().sort_values(ascending=False)
    rows = []
    for d in ds_med.index:
        sub = data[data["dataset"] == d]
        e_med = sub.groupby("edge")["cbdir"].median().sort_values(ascending=False)
        rows.extend((d, e) for e in e_med.index)
    return rows


def _brace_curve(n=201, beta=9.0):
    """Unit curly brace: t in [0, 1] along the span, h in [0, 1] with a central tip.

    Two logistic shoulders summed over each half give the classic brace profile;
    it is drawn as a polyline rather than Bezier segments so it stays correct
    under a blended transform, where the two axes have unrelated scales.
    """
    n = n if n % 2 else n + 1
    t = np.linspace(0.0, 1.0, n)
    half = t[: n // 2 + 1]
    s = (1.0 / (1.0 + np.exp(-2.0 * beta * (half - half[0])))
         + 1.0 / (1.0 + np.exp(-2.0 * beta * (half - half[-1]))))
    h = np.concatenate([s, s[-2::-1]])
    h = h - h.min()
    return t, h / max(h.max(), 1e-12)


def _draw_brace(ax, t0, t1, base, depth, transform, vertical=True, **kw):
    """Curly brace spanning t0..t1 on the category axis, tip `depth` past `base`.

    `vertical=True` means the categories run along x (the brace sits under the
    axis and opens upward); otherwise they run along y and the brace sits in the
    left margin. `base` and `depth` are in the transform's second coordinate.
    """
    t, h = _brace_curve()
    pos = t0 + t * (t1 - t0)
    off = base - depth * h
    xs, ys = (pos, off) if vertical else (off, pos)
    ax.plot(xs, ys, transform=transform, clip_on=False, **kw)
    return base - depth


def _fit_group_labels(fig, ax, groups, spans_px, plot_cfg, vertical):
    """Pick a rotation and font size at which no dataset name overlaps its neighbour.

    Measured, not guessed: each candidate is rendered once and its footprint along
    the category axis compared with the span of the group it labels. Rotating to
    90 degrees always fits (the footprint is then one line height), so the search
    terminates.
    """
    fs0 = _cfg_num(plot_cfg, "per_method_dataset_fontsize", 10.0)
    if vertical:
        cands = [(0, fs0), (0, fs0 * 0.85), (30, fs0), (45, fs0), (90, fs0 * 0.9)]
    else:
        cands = [(0, fs0), (0, fs0 * 0.85), (0, fs0 * 0.7)]
    try:
        rend = fig.canvas.get_renderer()
    except Exception:
        return cands[0][0], fs0
    for rot, fs in cands:
        ok = True
        for (lab, _a, _b), span in zip(groups, spans_px):
            t = ax.text(0, 0, lab, fontsize=fs, rotation=rot, fontweight="bold")
            bb = t.get_window_extent(renderer=rend)
            t.remove()
            foot = bb.width if vertical else bb.height
            if rot and vertical:
                r = np.deg2rad(rot)
                foot = bb.width * abs(np.cos(r)) + bb.height * abs(np.sin(r))
            if foot > max(span - 4.0, 1.0):
                ok = False
                break
        if ok:
            return rot, fs
    return cands[-1][0], cands[-1][1]


def _group_bounds(rows):
    """Contiguous (dataset, start, stop) blocks over an ordered (dataset, edge) list."""
    groups, prev = [], None
    for i, (d_name, _e) in enumerate(rows):
        if d_name != prev:
            groups.append([d_name, i, i + 1])
            prev = d_name
        else:
            groups[-1][2] = i + 1
    return [tuple(g) for g in groups]


def _annot_for(rows, tk, m, fields):
    """(text, colour, weight) per row from the per-edge test table."""
    out = []
    for d_name, e_name in rows:
        praw = padj = np.nan
        if tk is not None and (m, d_name, e_name) in tk.index:
            r = tk.loc[(m, d_name, e_name)]
            praw, padj = float(r["p"]), float(r["p_adj"])
        sig_adj = np.isfinite(padj) and padj < 0.05
        sig_raw = np.isfinite(praw) and praw < 0.05
        col, weight = ("#c2410c", "bold") if sig_adj else \
                      (("#b45309", "normal") if sig_raw else ("#777777", "normal"))
        bits = []
        if "p" in fields:
            bits.append(fmt_p(praw))
        if "p_adj" in fields:
            bits.append(fmt_p(padj, prefix="adj=") if "p" in fields else fmt_p(padj))
        out.append(("  ".join(bits), col, weight))
    return out


def plot_per_method(long_df, method_order, dataset_order, edges_by_dataset,
                    per_method_tests, plot_cfg, suffix):
    """One figure per method: every transition, the method against the reference.

    Vertical (default): x = transitions grouped by dataset, y = CBDir, with a
    curly brace under each dataset's block carrying the dataset name. Horizontal
    transposes that. Datasets are ordered by decreasing median CBDir and, within
    a dataset, the transitions likewise. Each transition carries its paired
    Wilcoxon p, corrected across the transitions of its own dataset.
    """
    ref = plot_cfg.get("reference_method")
    if not ref or ref not in method_order:
        print("  Skipping per-method panels: no reference_method in method_order")
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(save_base)),
                           str(plot_cfg.get("per_method_subdir", "per_method_plots")))
    stem = os.path.basename(save_base)
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    how = str(plot_cfg.get("per_method_order", "median")).lower()
    showfliers = bool(plot_cfg.get("per_method_showfliers",
                                   plot_cfg.get("show_points", False)))
    lw = plot_cfg.get("box_linewidth", 0.7)
    fields = plot_cfg.get("per_method_annotation", ["p", "p_adj"])
    if isinstance(fields, str):
        fields = [fields]
    fields = [str(f).strip().lower() for f in fields]

    tests = per_method_tests if per_method_tests is not None else pd.DataFrame()
    tk = (tests.set_index(["method", "dataset", "edge"])
          if not tests.empty else None)

    vertical = not str(plot_cfg.get("per_method_orientation", "vertical")
                       ).lower().startswith("h")
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory

    for m in method_order:
        if m == ref:
            continue
        data = long_df[long_df["method"].isin([ref, m])]
        if data.empty:
            continue
        rows = _per_method_row_order(data, ref, m, dataset_order, edges_by_dataset, how)
        if not rows:
            continue
        n = len(rows)
        groups = _group_bounds(rows)
        annots = _annot_for(rows, tk, m, fields)
        pair = [ref, m]                      # reference drawn first of the pair
        default_size = ([max(0.62 * n + 3.0, 7.0), 8.5] if vertical
                        else [11.0, max(0.42 * n + 2.4, 4.0)])
        figsize = tuple(plot_cfg.get("per_method_figsize") or default_size)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        half = 0.2

        for i, (d_name, e_name) in enumerate(rows):
            for k, mm in enumerate(pair):
                pos = i + (k - 0.5) * 2 * half
                vals = data.loc[(data["dataset"] == d_name) & (data["edge"] == e_name)
                                & (data["method"] == mm), "cbdir"].dropna().to_numpy()
                if vals.size == 0:
                    continue
                bp = ax.boxplot(
                    [vals], positions=[pos], widths=half * 1.6, vert=vertical,
                    patch_artist=True, showfliers=showfliers, manage_ticks=False,
                    boxprops=dict(linewidth=lw, edgecolor="#333333"),
                    whiskerprops=dict(linewidth=lw, color="#333333"),
                    capprops=dict(linewidth=lw, color="#333333"),
                    medianprops=dict(linewidth=lw * 1.6, color="#111111"),
                    flierprops=dict(marker=".", markersize=2, markeredgewidth=0.3,
                                    markerfacecolor="#666666", markeredgecolor="#666666"))
                for patch in bp["boxes"]:
                    patch.set_facecolor(palette.get(mm, "#888888"))
                    patch.set_alpha(0.85)

        lo_ax, hi_ax, _ = resolve_ylim(data["cbdir"], plot_cfg, annot_frac=0.0)
        span = hi_ax - lo_ax
        has_annot = any(t for t, _c, _w in annots)
        v_annot = hi_ax + 0.02 * span            # where the p-values start
        cat_lo, cat_hi = -0.5, n - 0.5
        vmax = hi_ax + (0.05 if vertical else 0.30) * span if has_annot else hi_ax
        labels = [e for _d, e in rows]
        rot_e = plot_cfg.get("per_method_edge_rotation")
        rot_e = (45 if vertical else 0) if rot_e is None else float(rot_e)

        if vertical:
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, fontsize=9, rotation=rot_e,
                               ha="right" if rot_e else "center",
                               rotation_mode="anchor" if rot_e else None)
            ax.set_xlim(cat_lo, cat_hi)
            ax.set_ylim(lo_ax, vmax)
        else:
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_ylim(cat_hi, cat_lo)
            ax.set_xlim(lo_ax, vmax)

        texts = []
        for i, (txt, col_t, weight) in enumerate(annots):
            if not txt:
                continue
            if vertical:
                texts.append(ax.text(i, v_annot, txt, ha="center", va="bottom",
                                     rotation=90, rotation_mode="anchor",
                                     fontsize=7.5, fontweight=weight, color=col_t))
            else:
                texts.append(ax.text(v_annot, i, txt, va="center", ha="left",
                                     fontsize=7.5, fontweight=weight, color=col_t))

        # alternating dataset bands and block separators
        band = ax.axvspan if vertical else ax.axhspan
        line = ax.axvline if vertical else ax.axhline
        for g, (_lab, a, b) in enumerate(groups):
            if g % 2 == 1:
                band(a - 0.5, b - 0.5, color="#000000", alpha=0.035, zorder=0)
            if a > 0:
                line(a - 0.5, color="#999999", lw=0.9, zorder=1)
        draw_zero_line(ax, plot_cfg, vertical=vertical)

        ticks = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(lo_ax, hi_ax)
        ticks = [t for t in ticks if lo_ax - 1e-9 <= t <= hi_ax + 1e-9]
        vlabel = plot_cfg.get("ylabel", VALUE_LABEL)
        if vertical:
            ax.set_yticks(ticks)
            ax.set_ylabel(vlabel, fontsize=12, fontweight="bold")
            ax.set_xlabel("")
        else:
            ax.set_xticks(ticks)
            ax.set_xlabel(vlabel, fontsize=12, fontweight="bold")
            ax.set_ylabel("")
        if has_annot:
            (ax.axhline if vertical else ax.axvline)(
                hi_ax, color="#bbbbbb", lw=0.8, zorder=1)

        ax.set_title(f"{m} vs {ref} — {VALUE_LABEL} by {UNIT_LABEL}\n"
                     f"per-{UNIT_LABEL} paired Wilcoxon over cells, corrected within dataset "
                     "(bold = adj < 0.05, amber = nominal only)",
                     fontsize=12, fontweight="bold", pad=12)
        handles = [Patch(facecolor=palette.get(x, "#888888"), edgecolor="#333333",
                         alpha=0.85, label=x) for x in pair]
        if vertical:
            ax.legend(handles=handles, title="Method", loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), frameon=True)
        else:
            ax.legend(handles=handles, title="Method", loc="upper center", ncol=2,
                      bbox_to_anchor=(0.5, -0.06 - 0.62 / max(float(figsize[1]), 1.0)),
                      frameon=True)
        fig.tight_layout()

        # The p-values are anchored in data coordinates, so grow the value axis
        # until they sit inside it rather than overprinting the title.
        if vertical and texts:
            for _ in range(3):
                try:
                    fig.canvas.draw()
                    rend = fig.canvas.get_renderer()
                    box = ax.get_window_extent(renderer=rend)
                    top = max(t.get_window_extent(renderer=rend).y1 for t in texts)
                    if top <= box.y1 - 2:
                        break
                    lo_cur, hi_cur = ax.get_ylim()
                    per_px = (hi_cur - lo_cur) / max(box.height, 1.0)
                    ax.set_ylim(lo_cur, hi_cur + (top - box.y1 + 6) * per_px)
                except Exception:
                    break
            fig.tight_layout()

        # Curly braces naming each dataset, measured off the rendered tick labels
        # so they clear the longest transition name instead of guessing a margin.
        if plot_cfg.get("per_method_group_braces", True):
            btrans = blended_transform_factory(*((ax.transData, ax.transAxes) if vertical
                                                 else (ax.transAxes, ax.transData)))
            extent, spans, along_px, across_px = 0.06, [1e9] * len(groups), 1.0, 400.0
            try:
                fig.canvas.draw()
                rend = fig.canvas.get_renderer()
                box = ax.get_window_extent(renderer=rend)
                ticklabs = ax.get_xticklabels() if vertical else ax.get_yticklabels()
                along_px = max(box.width if vertical else box.height, 1.0)
                across_px = max(box.height if vertical else box.width, 1.0)
                sizes = [t.get_window_extent(renderer=rend) for t in ticklabs]
                extent = max(((s.height if vertical else s.width) for s in sizes),
                             default=0.0) / across_px
                spans = [(b - a) / n * along_px for _l, a, b in groups]
            except Exception:
                pass
            depth = 9.0 / 72.0 * fig.dpi / across_px
            gap = 5.0 / 72.0 * fig.dpi / across_px
            base = -(extent + gap)
            rot_d, fs_d = _fit_group_labels(fig, ax, groups, spans, plot_cfg, vertical)
            for lab, a, b in groups:
                tip = _draw_brace(ax, a - 0.42, b - 1 + 0.42, base, depth, btrans,
                                  vertical=vertical, color="#555555", lw=1.1,
                                  solid_capstyle="round", zorder=5)
                mid = (a + b - 1) / 2.0
                if vertical:
                    ax.text(mid, tip - gap, lab, transform=btrans, ha="center",
                            va="top", rotation=rot_d,
                            rotation_mode="anchor" if rot_d else None,
                            fontsize=fs_d, fontweight="bold", color="#333333",
                            clip_on=False)
                else:
                    ax.text(tip - gap, mid, lab, transform=btrans, ha="right",
                            va="center", fontsize=fs_d, fontweight="bold",
                            color="#333333", clip_on=False)
        _save_figure(os.path.join(out_dir, f"{stem}_{_slug(m)}_vs_{_slug(ref)}{suffix}"), dpi)

    if not tests.empty:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        pth = os.path.join(out_dir, f"{stem}_per_edge_wilcoxon{suffix}.csv")
        tests.to_csv(pth, index=False)
        print(f"  Saved: {pth}  ({tests.shape[0]} rows)")


# ---------------------------------------------------------------------------
# Left-out cells: is the paired complete-case set representative?
# ---------------------------------------------------------------------------

def fit_leftout(long_df, method_order, plot_cfg):
    """Do the cells a paired comparison discards score like the cells it keeps?

    Every paired test in this module conditions on the cells the method and the
    reference both scored, which is only a fair restriction if the cells left out
    are exchangeable with the cells kept. This checks that assumption directly.

    For each (dataset, transition, method) the cells split three ways: scored by
    both, by the reference only, by the method only. The comparison is made
    **within one method's own scores** — the reference's unique cells against the
    reference's common cells, and the method's unique cells against its own common
    cells — so it asks purely about which cells got dropped, never about which
    method is better. A positive delta means the discarded cells scored HIGHER
    than the retained ones, i.e. the paired analysis is looking at a pessimistic
    subset for that method.

    Two layers, because the per-cell test is pseudoreplicated exactly like the
    others: per-transition Mann-Whitney (corrected across the transitions of its
    own dataset), and a transition-level summary that tests the 30 per-transition
    deltas, which is the layer any claim should rest on.

    A cell can be missing either because the method never scored it or because it
    had two or fewer target-type neighbours; with include_skipped off upstream the
    two are indistinguishable, so `n_target_neighbors_known` records whether the
    input could tell them apart rather than guessing a cause.
    """
    ref = plot_cfg.get("reference_method")
    if not ref or ref not in method_order:
        return pd.DataFrame(), pd.DataFrame()
    transform = str(plot_cfg.get("transform", "atanh")).lower()
    eps = _cfg_num(plot_cfg, "transform_eps", 1e-6)
    min_cells = _cfg_int(plot_cfg, "leftout_min_cells", 10)
    p_adjust = str(plot_cfg.get("leftout_p_adjust")
                   or plot_cfg.get("p_adjust", "fdr_bh")).lower()
    from statsmodels.stats.multitest import multipletests

    known = ("n_target_neighbors" in long_df.columns
             and long_df["cbdir"].isna().any())
    rows = []
    for m in method_order:
        if m == ref:
            continue
        sub = long_df[long_df["method"].isin([ref, m])]
        if sub.empty:
            continue
        for (d_name, e_name), esub in sub.groupby(["dataset", "edge"], sort=False):
            w = esub.pivot_table(index="obs_id", columns="method",
                                 values="cbdir", aggfunc="first")
            if ref not in w.columns or m not in w.columns:
                continue
            hr, hm = w[ref].notna(), w[m].notna()
            common, only_ref, only_m = hr & hm, hr & ~hm, hm & ~hr
            for side, col, mask in (("reference", ref, only_ref), ("method", m, only_m)):
                a = w.loc[common, col].dropna().to_numpy()      # kept cells
                b = w.loc[mask, col].dropna().to_numpy()        # discarded cells
                za, _ = _forward_transform(a, transform, eps)
                zb, _ = _forward_transform(b, transform, eps)
                rec = dict(method=m, reference_method=ref, dataset=d_name, edge=e_name,
                           side=side, scored_by=col, n_common=int(a.size),
                           n_unique=int(b.size),
                           frac_unique=float(b.size / max(a.size + b.size, 1)),
                           median_common=float(np.median(a)) if a.size else np.nan,
                           median_unique=float(np.median(b)) if b.size else np.nan,
                           delta_z=(float(zb.mean() - za.mean())
                                    if (a.size and b.size) else np.nan),
                           auc=np.nan, p=np.nan,
                           n_target_neighbors_known=bool(known))
                if a.size >= min_cells and b.size >= min_cells:
                    try:
                        u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
                        rec["auc"] = float(u) / (b.size * a.size)
                        rec["p"] = float(p)
                    except Exception as e:
                        print(f"    WARNING: left-out test failed "
                              f"({m}/{d_name}/{e_name}/{side}): {e}")
                rows.append(rec)

    per_edge = pd.DataFrame(rows)
    if per_edge.empty:
        return per_edge, pd.DataFrame()
    # correction family: the transitions of one dataset, for one method and side
    per_edge["p_adj"] = np.nan
    per_edge["n_edges_in_dataset"] = 0
    per_edge["p_adjust"] = p_adjust
    for _key, idx in per_edge.groupby(["method", "side", "dataset"], sort=False).groups.items():
        v = per_edge.loc[idx, "p"].to_numpy(dtype=float)
        per_edge.loc[idx, "n_edges_in_dataset"] = int(v.size)
        ok = np.where(np.isfinite(v))[0]
        if not ok.size:
            continue
        adj = np.full(v.size, np.nan)
        adj[ok] = (multipletests(v[ok], method=p_adjust)[1] if p_adjust != "none" else v[ok])
        per_edge.loc[idx, "p_adj"] = adj

    # transition-level summary: test the per-transition deltas, not the cells
    srows = []
    for (m, side), g in per_edge.groupby(["method", "side"], sort=False):
        d = g["delta_z"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        rec = dict(method=m, reference_method=ref, side=side, n_edges=int(d.size),
                   mean_delta_z=float(d.mean()) if d.size else np.nan,
                   mean_auc=float(np.nanmean(g["auc"])) if g["auc"].notna().any() else np.nan,
                   mean_frac_unique=float(g["frac_unique"].mean()),
                   n_edges_higher=int((d > 0).sum()))
        if d.size >= 3:
            se = d.std(ddof=1) / np.sqrt(d.size)
            tc = stats.t.ppf(0.975, d.size - 1)
            rec.update(se=float(se), ci_low=float(d.mean() - tc * se),
                       ci_high=float(d.mean() + tc * se),
                       p_edge=float(stats.ttest_1samp(d, 0).pvalue))
            try:
                rec["p_signrank"] = float(stats.wilcoxon(d).pvalue)
            except Exception:
                rec["p_signrank"] = np.nan
        else:
            rec.update(se=np.nan, ci_low=np.nan, ci_high=np.nan,
                       p_edge=np.nan, p_signrank=np.nan)
        srows.append(rec)
    summary = pd.DataFrame(srows)
    for side, g in summary.groupby("side", sort=False):
        v = g["p_edge"].to_numpy(dtype=float)
        ok = np.where(np.isfinite(v))[0]
        adj = np.full(v.size, np.nan)
        if ok.size:
            adj[ok] = (multipletests(v[ok], method=p_adjust)[1]
                       if p_adjust != "none" else v[ok])
        summary.loc[g.index, "p_edge_adj"] = adj

    print(f"\n=== Left-out cells vs retained cells ({UNIT_LABEL} level) ===")
    print("    delta > 0 means the discarded cells scored HIGHER than the kept ones")
    for _, r in summary.iterrows():
        print(f"  {r['method']:<20s} {r['side']:<9s} "
              f"dropped {r['mean_frac_unique']*100:5.1f}%  "
              f"delta_z={r['mean_delta_z']:+.4f}  "
              f"higher on {r['n_edges_higher']}/{r['n_edges']} transitions  "
              f"{fmt_p(r['p_edge'])} {fmt_p(r.get('p_edge_adj', np.nan), prefix='adj=')}")
    return per_edge, summary


def plot_leftout_summary(summary, method_order, plot_cfg, suffix):
    """Transition-level forest: do the discarded cells score higher or lower?"""
    if summary is None or summary.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    ref = summary["reference_method"].iloc[0]
    sides = [s for s in ("reference", "method") if (summary["side"] == s).any()]
    present = [m for m in method_order if m in set(summary["method"])][::-1]
    if not present or not sides:
        return
    figsize = tuple(plot_cfg.get("leftout_summary_figsize")
                    or [5.6 * len(sides), 0.52 * max(len(present), 3) + 2.6])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(sides), figsize=figsize, dpi=dpi,
                             squeeze=False, sharey=True)
    titles = {"reference": f"Cells only {ref} scored\n(vs the cells it shares with the method)",
              "method": "Cells only the method scored\n(vs the cells it shares with "
                        f"{ref})"}
    for ax, side in zip(axes[0], sides):
        g = summary[summary["side"] == side].set_index("method")
        for y, m in enumerate(present):
            if m not in g.index:
                continue
            r = g.loc[m]
            est, lo, hi = r["mean_delta_z"], r.get("ci_low"), r.get("ci_high")
            if not np.isfinite(est):
                ax.text(0.0, y, "  not estimable", va="center", fontsize=8, color="#999999")
                continue
            col = palette.get(m, "#444444")
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot([est], [y], "o", color=col, ms=7, zorder=3,
                    markeredgecolor="#333333", markeredgewidth=0.5)
            praw, padj = r.get("p_edge", np.nan), r.get("p_edge_adj", np.nan)
            sig_adj = np.isfinite(padj) and padj < 0.05
            sig_raw = np.isfinite(praw) and praw < 0.05
            col_t, weight = ("#c2410c", "bold") if sig_adj else \
                            (("#b45309", "normal") if sig_raw else ("#777777", "normal"))
            x = hi if np.isfinite(hi) else est
            ax.text(x, y, f"   {fmt_p(praw)}  {fmt_p(padj, prefix='adj=')}"
                          f"   ({r['mean_frac_unique']*100:.0f}% dropped)",
                    va="center", fontsize=7.5, fontweight=weight, color=col_t)
        draw_zero_line(ax, plot_cfg, vertical=False)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present)
        ax.set_xlabel(f"mean {VALUE_LABEL}(left-out) − {VALUE_LABEL}(retained), Fisher z",
                      fontsize=10)
        ax.set_title(titles.get(side, side), fontsize=11, fontweight="bold")
        lo_x, hi_x = ax.get_xlim()
        ax.set_xlim(lo_x, hi_x + 0.55 * (hi_x - lo_x))
    fig.suptitle("Are the cells a paired comparison discards like the ones it keeps?"
                 "\n0 = exchangeable, >0 = the discarded cells scored higher"
                 "   |   one point per method, 95% CI over transitions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save_figure(f"{save_base}_leftout{suffix}", dpi)


def _draw_cat_boxes(ax, values, n, colors, hatches, vertical, lw, showfliers, width=0.84):
    """k series of boxes at each of n category slots; returns the slot half-width."""
    k = max(len(values), 1)
    bw = width / k
    for s in range(k):
        for i in range(n):
            vals = values[s][i]
            if vals is None or len(vals) == 0:
                continue
            pos = i - width / 2.0 + (s + 0.5) * bw
            bp = ax.boxplot(
                [vals], positions=[pos], widths=bw * 0.86, vert=vertical,
                patch_artist=True, showfliers=showfliers, manage_ticks=False,
                boxprops=dict(linewidth=lw, edgecolor="#333333"),
                whiskerprops=dict(linewidth=lw, color="#333333"),
                capprops=dict(linewidth=lw, color="#333333"),
                medianprops=dict(linewidth=lw * 1.6, color="#111111"),
                flierprops=dict(marker=".", markersize=2, markeredgewidth=0.3,
                                markerfacecolor="#666666", markeredgecolor="#666666"))
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[s])
                patch.set_alpha(0.85)
                if hatches[s]:
                    patch.set_hatch(hatches[s])
    return bw


def plot_leftout_by_method(long_df, method_order, dataset_order, edges_by_dataset,
                           per_edge, plot_cfg, suffix):
    """Per method: retained vs left-out cells at every transition, both sides.

    Four boxes per transition — the reference's retained and left-out cells, then
    the method's — so the two splits can be read against each other. Hatched boxes
    are the left-out cells. The p above each pair is that side's Mann-Whitney,
    corrected across the transitions of its dataset.
    """
    ref = plot_cfg.get("reference_method")
    if not ref or per_edge is None or per_edge.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(save_base)),
                           str(plot_cfg.get("leftout_subdir", "leftout_plots")))
    stem = os.path.basename(save_base)
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    how = str(plot_cfg.get("per_method_order", "median")).lower()
    lw = plot_cfg.get("box_linewidth", 0.7)
    showfliers = bool(plot_cfg.get("per_method_showfliers",
                                   plot_cfg.get("show_points", False)))
    field = str(plot_cfg.get("leftout_annotation", "p_adj")).strip().lower()
    vertical = not str(plot_cfg.get("per_method_orientation", "vertical")
                       ).lower().startswith("h")
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory

    pk = per_edge.set_index(["method", "side", "dataset", "edge"])
    for m in method_order:
        if m == ref:
            continue
        data = long_df[long_df["method"].isin([ref, m])]
        if data.empty:
            continue
        rows = _per_method_row_order(data, ref, m, dataset_order, edges_by_dataset, how)
        if not rows:
            continue
        n = len(rows)
        groups = _group_bounds(rows)
        # four series: ref-retained, ref-left-out, method-retained, method-left-out
        series, labels_s, colors, hatches = [[], [], [], []], [], [], []
        for side, col in (("reference", ref), ("method", m)):
            for kind in ("retained", "left-out"):
                labels_s.append(f"{col} — {kind}")
                colors.append(palette.get(col, "#888888"))
                hatches.append("///" if kind == "left-out" else "")
        for i, (d_name, e_name) in enumerate(rows):
            esub = data[(data["dataset"] == d_name) & (data["edge"] == e_name)]
            w = esub.pivot_table(index="obs_id", columns="method",
                                 values="cbdir", aggfunc="first")
            hr = w[ref].notna() if ref in w.columns else pd.Series(False, index=w.index)
            hm = w[m].notna() if m in w.columns else pd.Series(False, index=w.index)
            common = hr & hm
            packs = [(ref, common), (ref, hr & ~hm), (m, common), (m, hm & ~hr)]
            for s, (col, mask) in enumerate(packs):
                v = (w.loc[mask, col].dropna().to_numpy()
                     if col in w.columns else np.array([]))
                series[s].append(v)
        allv = np.concatenate([v for s in series for v in s if len(v)]) \
            if any(len(v) for s in series for v in s) else np.array([0.0])
        lo_ax, hi_ax, _ = resolve_ylim(pd.Series(allv), plot_cfg, annot_frac=0.0)

        default_size = ([max(0.95 * n + 3.2, 8.0), 8.5] if vertical
                        else [12.0, max(0.62 * n + 2.6, 4.5)])
        figsize = tuple(plot_cfg.get("leftout_figsize") or default_size)
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        bw = _draw_cat_boxes(ax, series, n, colors, hatches, vertical, lw, showfliers)

        v_annot = hi_ax + 0.02 * (hi_ax - lo_ax)
        labels = [e for _d, e in rows]
        rot_e = plot_cfg.get("per_method_edge_rotation")
        rot_e = (45 if vertical else 0) if rot_e is None else float(rot_e)
        if vertical:
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, fontsize=9, rotation=rot_e,
                               ha="right" if rot_e else "center",
                               rotation_mode="anchor" if rot_e else None)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(lo_ax, hi_ax + 0.05 * (hi_ax - lo_ax))
        else:
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_xlim(lo_ax, hi_ax + 0.30 * (hi_ax - lo_ax))

        texts = []
        for i, (d_name, e_name) in enumerate(rows):
            for side, off in (("reference", -0.21), ("method", 0.21)):
                key = (m, side, d_name, e_name)
                if key not in pk.index:
                    continue
                r = pk.loc[key]
                praw, padj = r.get("p", np.nan), r.get("p_adj", np.nan)
                val = padj if field == "p_adj" else praw
                if not np.isfinite(val):
                    continue
                sig_adj = np.isfinite(padj) and padj < 0.05
                sig_raw = np.isfinite(praw) and praw < 0.05
                col_t, weight = ("#c2410c", "bold") if sig_adj else \
                                (("#b45309", "normal") if sig_raw else ("#999999", "normal"))
                txt = fmt_p(val, prefix="")
                if vertical:
                    texts.append(ax.text(i + off, v_annot, txt, ha="center", va="bottom",
                                         rotation=90, rotation_mode="anchor",
                                         fontsize=6.5, fontweight=weight, color=col_t))
                else:
                    texts.append(ax.text(v_annot, i + off, txt, ha="left", va="center",
                                         fontsize=6.5, fontweight=weight, color=col_t))

        band = ax.axvspan if vertical else ax.axhspan
        line = ax.axvline if vertical else ax.axhline
        for g, (_lab, a, b) in enumerate(groups):
            if g % 2 == 1:
                band(a - 0.5, b - 0.5, color="#000000", alpha=0.035, zorder=0)
            if a > 0:
                line(a - 0.5, color="#999999", lw=0.9, zorder=1)
        draw_zero_line(ax, plot_cfg, vertical=vertical)
        ticks = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(lo_ax, hi_ax)
        ticks = [t for t in ticks if lo_ax - 1e-9 <= t <= hi_ax + 1e-9]
        vlabel = plot_cfg.get("ylabel", VALUE_LABEL)
        if vertical:
            ax.set_yticks(ticks); ax.set_ylabel(vlabel, fontsize=12, fontweight="bold")
        else:
            ax.set_xticks(ticks); ax.set_xlabel(vlabel, fontsize=12, fontweight="bold")
        (ax.axhline if vertical else ax.axvline)(hi_ax, color="#bbbbbb", lw=0.8, zorder=1)

        ax.set_title(f"{m} vs {ref} — cells retained by the paired comparison vs cells left out"
                     "\nhatched = left out; p is that side's Mann-Whitney, corrected within "
                     "dataset (bold = adj < 0.05)",
                     fontsize=12, fontweight="bold", pad=12)
        handles = [Patch(facecolor=c, edgecolor="#333333", alpha=0.85, hatch=h, label=l)
                   for c, h, l in zip(colors, hatches, labels_s)]
        if vertical:
            ax.legend(handles=handles, title="Cells", loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), frameon=True)
        else:
            ax.legend(handles=handles, title="Cells", loc="upper center", ncol=2,
                      bbox_to_anchor=(0.5, -0.06 - 0.62 / max(float(figsize[1]), 1.0)),
                      frameon=True)
        fig.tight_layout()

        if vertical and texts:
            for _ in range(3):
                try:
                    fig.canvas.draw()
                    rend = fig.canvas.get_renderer()
                    box = ax.get_window_extent(renderer=rend)
                    top = max(t.get_window_extent(renderer=rend).y1 for t in texts)
                    if top <= box.y1 - 2:
                        break
                    lo_c, hi_c = ax.get_ylim()
                    ax.set_ylim(lo_c, hi_c + (top - box.y1 + 6) * (hi_c - lo_c)
                                / max(box.height, 1.0))
                except Exception:
                    break
            fig.tight_layout()

        if plot_cfg.get("per_method_group_braces", True):
            btrans = blended_transform_factory(*((ax.transData, ax.transAxes) if vertical
                                                 else (ax.transAxes, ax.transData)))
            extent, spans, across_px = 0.06, [1e9] * len(groups), 400.0
            try:
                fig.canvas.draw()
                rend = fig.canvas.get_renderer()
                box = ax.get_window_extent(renderer=rend)
                tl = ax.get_xticklabels() if vertical else ax.get_yticklabels()
                along_px = max(box.width if vertical else box.height, 1.0)
                across_px = max(box.height if vertical else box.width, 1.0)
                sz = [t.get_window_extent(renderer=rend) for t in tl]
                extent = max(((s.height if vertical else s.width) for s in sz),
                             default=0.0) / across_px
                spans = [(b - a) / n * along_px for _l, a, b in groups]
            except Exception:
                pass
            depth = 9.0 / 72.0 * fig.dpi / across_px
            gap = 5.0 / 72.0 * fig.dpi / across_px
            base = -(extent + gap)
            rot_d, fs_d = _fit_group_labels(fig, ax, groups, spans, plot_cfg, vertical)
            for lab, a, b in groups:
                tip = _draw_brace(ax, a - 0.42, b - 1 + 0.42, base, depth, btrans,
                                  vertical=vertical, color="#555555", lw=1.1,
                                  solid_capstyle="round", zorder=5)
                mid = (a + b - 1) / 2.0
                if vertical:
                    ax.text(mid, tip - gap, lab, transform=btrans, ha="center", va="top",
                            rotation=rot_d, rotation_mode="anchor" if rot_d else None,
                            fontsize=fs_d, fontweight="bold", color="#333333", clip_on=False)
                else:
                    ax.text(tip - gap, mid, lab, transform=btrans, ha="right", va="center",
                            fontsize=fs_d, fontweight="bold", color="#333333", clip_on=False)
        _save_figure(os.path.join(out_dir, f"{stem}_{_slug(m)}_leftout{suffix}"), dpi)


def plot_leftout_coverage(per_edge, method_order, plot_cfg, suffix):
    """Does the bias grow with how much the paired restriction throws away?"""
    if per_edge is None or per_edge.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    ref = per_edge["reference_method"].iloc[0]
    sides = [s for s in ("reference", "method") if (per_edge["side"] == s).any()]
    datasets = list(pd.unique(per_edge["dataset"]))
    marks = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    mk = {d: marks[i % len(marks)] for i, d in enumerate(datasets)}
    figsize = tuple(plot_cfg.get("leftout_coverage_figsize") or [5.8 * len(sides), 5.0])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(sides), figsize=figsize, dpi=dpi,
                             squeeze=False, sharey=True)
    for ax, side in zip(axes[0], sides):
        g = per_edge[(per_edge["side"] == side) & per_edge["delta_z"].notna()]
        for _, r in g.iterrows():
            ax.plot([100 * r["frac_unique"]], [r["delta_z"]], mk.get(r["dataset"], "o"),
                    color=palette.get(r["method"], "#888888"), ms=5.5, alpha=0.85,
                    markeredgecolor="#333333", markeredgewidth=0.4)
        x, y = 100 * g["frac_unique"].to_numpy(), g["delta_z"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 2 and np.unique(x[ok]).size > 1:
            sl = stats.linregress(x[ok], y[ok])
            xs = np.linspace(x[ok].min(), x[ok].max(), 50)
            ax.plot(xs, sl.intercept + sl.slope * xs, "--", color="#444444", lw=1.2)
            ax.text(0.02, 0.02, f"slope={sl.slope:+.4f} per 1% dropped\n"
                                f"r={sl.rvalue:+.2f}  {fmt_p(sl.pvalue)}",
                    transform=ax.transAxes, fontsize=8, va="bottom", color="#333333")
        draw_zero_line(ax, plot_cfg)
        ax.set_xlabel(f"% of the {UNIT_LABEL}'s cells left out", fontsize=10)
        ax.set_title(f"{'the reference' if side == 'reference' else 'the method'}'s "
                     "left-out cells", fontsize=11, fontweight="bold")
    axes[0][0].set_ylabel(f"{VALUE_LABEL}(left-out) − {VALUE_LABEL}(retained), Fisher z",
                          fontsize=11, fontweight="bold")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=palette.get(m, "#888888"), label=m)
         for m in method_order if m in set(per_edge["method"])]
    h += [Line2D([], [], marker=mk[d], ls="", color="#555555", label=d) for d in datasets]
    axes[0][-1].legend(handles=h, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left",
                       frameon=True, title="method / dataset")
    fig.suptitle("Is the paired restriction informative?"
                 "\na non-zero slope means the more cells a comparison drops, "
                 "the more biased the kept set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save_figure(f"{save_base}_leftout_coverage{suffix}", dpi)


# ---------------------------------------------------------------------------
# Stability: level and consistency together, with no reference method
# ---------------------------------------------------------------------------

def build_method_levels(long_df, transform="atanh", eps=1e-6):
    """(dataset, transition) x method matrix of each method's own-cell mean, in z.

    Reference-free: every method is summarised on the cells it actually scored,
    so the matrix supports statements about all methods at once rather than about
    offsets from one of them. This is the primitive the consistency claims need —
    "which method never fails a dataset" is not expressible as a set of pairwise
    offsets, because each offset is computed on a different subset of cells.
    """
    d = long_df.dropna(subset=["cbdir"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    z, _ = _forward_transform(d["cbdir"].to_numpy(), transform, eps)
    d = d.assign(_z=z)
    g = d.groupby(["dataset", "edge", "method"])["_z"]
    levels = g.mean().unstack("method")
    counts = g.size().unstack("method")
    return levels, counts


def _tau_between_dataset(levels_col):
    """Between-dataset SD of one method's transition means, and the within SD.

    A plain SD across dataset means confounds real dataset-to-dataset
    heterogeneity with the sampling noise of estimating each dataset mean from a
    handful of transitions; the variance components separate them.
    """
    sub = levels_col.dropna().reset_index()
    sub.columns = ["dataset", "edge", "y"]
    if sub["dataset"].nunique() < 3 or len(sub) < 5:
        return np.nan, np.nan
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.mixedlm("y ~ 1", sub, groups=sub["dataset"]).fit(reml=True)
        return float(np.sqrt(max(float(f.cov_re.iloc[0, 0]), 0.0))), float(np.sqrt(f.scale))
    except Exception:
        return np.nan, np.nan


def fit_stability(long_df, method_order, plot_cfg):
    """Level and consistency per method, plus a dataset cluster bootstrap.

    Spread on its own is a trap: a method that is uniformly mediocre has a small
    SD and a low variance component while being the worst method in the panel. So
    every dispersion statistic here is reported beside a level statistic, and the
    headline endpoint is the **worst-dataset mean** — one number that cannot be
    won by being uniformly bad, and that states the claim ("it works everywhere")
    directly.
    """
    transform = str(plot_cfg.get("transform", "atanh")).lower()
    eps = _cfg_num(plot_cfg, "transform_eps", 1e-6)
    levels, counts = build_method_levels(long_df, transform, eps)
    if levels.empty:
        return pd.DataFrame(), levels
    methods = [m for m in method_order if m in levels.columns]
    if not methods:
        methods = list(levels.columns)
    levels = levels[methods]
    ds_means = levels.groupby(level="dataset").mean()
    n_ds = ds_means.shape[0]

    rows = []
    for m in methods:
        col, dsm = levels[m].dropna(), ds_means[m].dropna()
        tau, sig = _tau_between_dataset(levels[[m]])
        n_pos = int((col > 0).sum())
        rec = dict(
            method=m, n_transitions=int(col.size), n_datasets=int(dsm.size),
            mean_z=float(col.mean()), median_z=float(col.median()),
            iqr_z=float(col.quantile(0.75) - col.quantile(0.25)),
            sd_transitions=float(col.std(ddof=1)),
            mean_of_dataset_means=float(dsm.mean()),
            sd_datasets=float(dsm.std(ddof=1)) if dsm.size > 1 else np.nan,
            worst_dataset=float(dsm.min()), best_dataset=float(dsm.max()),
            worst_dataset_name=str(dsm.idxmin()) if dsm.size else "",
            worst_transition=float(col.min()),
            n_datasets_positive=int((dsm > 0).sum()),
            n_transitions_positive=n_pos,
            p_transitions_positive=float(
                stats.binomtest(n_pos, int(col.size), 0.5, "greater").pvalue)
            if col.size else np.nan,
            tau_between_dataset=tau, sigma_within=sig,
            icc_dataset=(tau ** 2 / (tau ** 2 + sig ** 2)
                         if np.isfinite(tau) and np.isfinite(sig) and (tau or sig) else np.nan),
            n_cells=int(counts[m].sum()) if m in counts.columns else 0)
        rec["mean_minus_sd"] = (rec["mean_of_dataset_means"] - rec["sd_datasets"]
                                if np.isfinite(rec["sd_datasets"]) else np.nan)
        for src, dst in (("mean_z", "mean_native"), ("median_z", "median_native"),
                         ("worst_dataset", "worst_dataset_native"),
                         ("best_dataset", "best_dataset_native"),
                         ("worst_transition", "worst_transition_native"),
                         ("mean_of_dataset_means", "mean_of_dataset_means_native")):
            rec[dst] = float(_inverse_transform(np.array([rec[src]]), transform)[0])
        rows.append(rec)
    out = pd.DataFrame(rows)

    B = _cfg_int(plot_cfg, "stability_bootstrap", 5000)
    if B > 0 and n_ds > 1:
        rng = np.random.default_rng(_cfg_int(plot_cfg, "stability_seed", 0))
        by = {d: levels.xs(d, level="dataset") for d in ds_means.index}
        names = list(ds_means.index)
        win_worst = {m: 0 for m in methods}
        win_ms = {m: 0 for m in methods}
        win_sd = {m: 0 for m in methods}
        for _ in range(B):
            pick = rng.integers(0, n_ds, n_ds)
            dm = pd.DataFrame([by[names[i]].mean() for i in pick])
            mn, sd = dm.min(), dm.std(ddof=1)
            win_worst[mn.idxmax()] += 1
            win_ms[(dm.mean() - sd).idxmax()] += 1
            win_sd[sd.idxmin()] += 1
        out["p_best_worst_dataset"] = out["method"].map(lambda m: win_worst[m] / B)
        out["p_best_mean_minus_sd"] = out["method"].map(lambda m: win_ms[m] / B)
        out["p_smallest_sd"] = out["method"].map(lambda m: win_sd[m] / B)
        out["n_bootstrap"] = B

    vs_ref = stability_vs_reference(levels, methods, plot_cfg)
    if not vs_ref.empty:
        out = out.merge(vs_ref, on="method", how="left")

    print("\n=== Level and consistency (each method on its own cells) ===")
    print("    worst-dataset mean is the headline: a uniformly mediocre method "
          "cannot win it")
    for _, r in out.sort_values("worst_dataset", ascending=False).iterrows():
        print(f"  {r['method']:<20s} mean={r['mean_z']:+.4f}  "
              f"sd(datasets)={r['sd_datasets']:.4f}  tau={r['tau_between_dataset']:.4f}  "
              f"worst dataset={r['worst_dataset']:+.4f} ({r['worst_dataset_name']})  "
              f"positive on {int(r['n_datasets_positive'])}/{int(r['n_datasets'])} datasets, "
              f"{int(r['n_transitions_positive'])}/{int(r['n_transitions'])} "
              f"{UNIT_LABEL_PLURAL}")
    return out, levels


def stability_vs_reference(levels, methods, plot_cfg):
    """Paired Wilcoxon of each method against the reference, dataset by dataset.

    The unit of replication is the DATASET, matching what the left stability
    panel draws: one value per dataset per method (the mean over that dataset's
    transitions, on the Fisher-z scale), paired across methods because every
    method is scored on the same datasets. This is deliberately a different and
    much more conservative question than the pooled cell-level tests elsewhere in
    the pipeline — with a handful of datasets the smallest attainable two-sided
    p is 2 / 2**n_datasets, so a non-significant result here is a statement about
    how many datasets there are, not evidence of equivalence.

    Note that each method's per-dataset mean is computed on the cells that method
    itself scored, so this pairs dataset-level summaries rather than cells; it is
    the reference-free `levels` matrix, not the paired-cell subset.

    p-values are FDR-adjusted across the non-reference methods (`p_adjust`).
    """
    ref = plot_cfg.get("reference_method")
    if levels is None or levels.empty or not ref or ref not in levels.columns:
        if ref and (levels is not None) and not levels.empty:
            print(f"  Skipping stability-vs-reference test: reference_method "
                  f"'{ref}' is not among the scored methods")
        return pd.DataFrame()

    alt = str(plot_cfg.get("stability_wilcoxon_alternative", "two-sided")).lower()
    zero_method = str(plot_cfg.get("stability_wilcoxon_zero_method", "wilcox"))
    p_adjust = str(plot_cfg.get("stability_p_adjust")
                   or plot_cfg.get("p_adjust", "fdr_bh")).lower()
    min_ds = _cfg_int(plot_cfg, "stability_wilcoxon_min_datasets", 3)

    ds_means = levels.groupby(level="dataset").mean()
    rows = []
    for m in methods:
        if m == ref:
            rows.append(dict(method=m, reference_method=ref, n_datasets_paired=np.nan,
                             median_diff_vs_ref=np.nan, w_stat_vs_ref=np.nan,
                             p_wilcoxon_vs_ref=np.nan))
            continue
        pair = ds_means[[m, ref]].dropna()
        d = (pair[m] - pair[ref]).to_numpy(dtype=float)
        rec = dict(method=m, reference_method=ref, n_datasets_paired=int(d.size),
                   median_diff_vs_ref=float(np.median(d)) if d.size else np.nan,
                   w_stat_vs_ref=np.nan, p_wilcoxon_vs_ref=np.nan)
        if d.size >= min_ds and np.any(d != 0):
            try:
                w, p = stats.wilcoxon(pair[m].to_numpy(dtype=float),
                                      pair[ref].to_numpy(dtype=float),
                                      alternative=alt, zero_method=zero_method)
                rec["w_stat_vs_ref"], rec["p_wilcoxon_vs_ref"] = float(w), float(p)
            except Exception as e:
                print(f"  WARNING: stability Wilcoxon failed for {m}: "
                      f"{type(e).__name__}: {e}")
        rows.append(rec)

    out = pd.DataFrame(rows)
    p = out["p_wilcoxon_vs_ref"].to_numpy(dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        if p_adjust == "none":
            adj[ok] = p[ok]
        else:
            from statsmodels.stats.multitest import multipletests
            adj[ok] = multipletests(p[ok], method=p_adjust)[1]
    out["p_wilcoxon_vs_ref_adj"] = adj
    out["stability_p_adjust"] = p_adjust
    out["stability_wilcoxon_alternative"] = alt

    print(f"\n=== Paired Wilcoxon across datasets vs {ref} "
          f"(unit = dataset, {alt}, {p_adjust}) ===")
    for _, r in out.iterrows():
        if r["method"] == ref:
            print(f"  {r['method']:<20s} reference")
        elif np.isfinite(r["p_wilcoxon_vs_ref"]):
            print(f"  {r['method']:<20s} n={int(r['n_datasets_paired'])} datasets  "
                  f"median diff={r['median_diff_vs_ref']:+.4f}  "
                  f"p={r['p_wilcoxon_vs_ref']:.3g}  "
                  f"p_adj={r['p_wilcoxon_vs_ref_adj']:.3g}")
        else:
            print(f"  {r['method']:<20s} not testable "
                  f"(n={r['n_datasets_paired']} paired datasets)")
    return out


def fit_head_to_head(long_df, plot_cfg):
    """A pre-specified two-method comparison, stated as a comparison of classes.

    This is the one contrast that does not need a multiplicity correction and
    does not inherit the winner's-curse problem the eight-method panel has: the
    pair is fixed before the data are seen, on a structural criterion (here, the
    two methods that do not use spliced/unspliced kinetics), so it is a single
    a-priori hypothesis rather than the largest of eight observed gaps.

    Everything is computed on the same nesting the rest of the pipeline uses:
    per-cell values collapse to a transition, transitions to a dataset, and the
    honest p comes from a dataset-level sign flip whose floor is 1 / 2**n_datasets.
    Four different questions are reported because they fail differently — a shift
    test assumes a location shift, a sign test throws away magnitude, and the
    reversal count assumes nothing about shape but only sees direction.
    """
    pair = plot_cfg.get("head_to_head")
    if not pair or len(list(pair)) != 2:
        return pd.DataFrame(), pd.DataFrame(), {}
    focus, comp = str(pair[0]), str(pair[1])
    d = long_df.dropna(subset=["cbdir"])
    have = set(d["method"])
    if focus not in have or comp not in have:
        print(f"  Skipping the head-to-head: {focus} vs {comp} — "
              f"{[m for m in (focus, comp) if m not in have]} not in the run")
        return pd.DataFrame(), pd.DataFrame(), {}

    transform = str(plot_cfg.get("transform", "atanh")).lower()
    eps = _cfg_num(plot_cfg, "transform_eps", 1e-6)
    z, _ = _forward_transform(d["cbdir"].to_numpy(), transform, eps)
    d = d.assign(_z=z)

    # Per transition, on each method's OWN scored cells — the reference-free
    # summary, so neither method is measured on a subset chosen by the other.
    per_edge = (d.groupby(["dataset", "edge", "method"])["_z"].mean()
                  .unstack("method"))
    per_edge = per_edge[[c for c in (focus, comp) if c in per_edge.columns]].dropna()
    if per_edge.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    per_edge = per_edge.rename(columns={focus: "focus", comp: "comparator"})
    per_edge["diff"] = per_edge["focus"] - per_edge["comparator"]
    per_edge = per_edge.reset_index()
    per_ds = (per_edge.groupby("dataset")[["focus", "comparator", "diff"]]
                      .median().reset_index())

    x = per_edge["diff"].to_numpy(dtype=float)
    n_t, n_d = int(x.size), int(per_ds.shape[0])
    st = dict(focus=focus, comparator=comp, n_transitions=n_t, n_datasets=n_d,
              n_transitions_focus_higher=int((x > 0).sum()),
              n_datasets_focus_higher=int((per_ds["diff"] > 0).sum()),
              median_diff=float(np.median(x)), mean_diff=float(np.mean(x)))

    w = np.sort(np.array([(x[i] + x[j]) / 2 for i in range(n_t) for j in range(i, n_t)]))
    st["hl_shift"] = float(np.median(w))
    k = int(np.floor(n_t * (n_t + 1) / 4
                     - 1.96 * np.sqrt(n_t * (n_t + 1) * (2 * n_t + 1) / 24)))
    k = max(k, 0)
    st["hl_ci_low"], st["hl_ci_high"] = float(w[k]), float(w[len(w) - 1 - k])
    try:
        st["p_signrank_one_sided"] = float(
            stats.wilcoxon(x, alternative="greater").pvalue)
    except Exception:
        st["p_signrank_one_sided"] = np.nan
    st["p_sign_one_sided"] = float(
        stats.binomtest(int((x > 0).sum()), n_t, 0.5, alternative="greater").pvalue)

    # Dataset-level sign flip: exact, and its floor is worth quoting whenever the
    # p is reported, because no amount of effect can push it below 1 / 2**n.
    dm = per_ds["diff"].to_numpy(dtype=float)
    signs = np.array(list(itertools.product([1, -1], repeat=n_d)))
    null = (signs * dm).mean(axis=1)
    st["p_signflip_one_sided"] = float((null >= dm.mean() - 1e-12).mean())
    st["signflip_floor"] = float(1.0 / 2 ** n_d)

    # Direction correctness per transition: assumes nothing about shape, and is
    # the endpoint a bimodal comparator fails on while its median looks mild.
    a_pos, b_pos = per_edge["focus"] > 0, per_edge["comparator"] > 0
    b_only = int((a_pos & ~b_pos).sum())
    c_only = int((~a_pos & b_pos).sum())
    st.update(both_correct=int((a_pos & b_pos).sum()), only_focus_correct=b_only,
              only_comparator_correct=c_only,
              both_reversed=int((~a_pos & ~b_pos).sum()),
              p_mcnemar=float(stats.binomtest(c_only, b_only + c_only, 0.5,
                                              alternative="less").pvalue)
              if (b_only + c_only) else np.nan)

    # Cell-level win rate on the cells both scored, with a dataset cluster CI.
    if "obs_id" in d.columns:
        w_rows = []
        for ds_name, g in d[d["method"].isin([focus, comp])].groupby("dataset"):
            piv = g.pivot_table(index="obs_id", columns="method", values="_z")
            if {focus, comp}.issubset(piv.columns):
                piv = piv.dropna()
                if len(piv):
                    w_rows.append((ds_name, float((piv[focus] > piv[comp]).mean()),
                                   int(len(piv))))
        if w_rows:
            wv = np.array([r[1] for r in w_rows])
            st["p_win_cells"] = float(np.mean(wv))
            rng = np.random.default_rng(_cfg_int(plot_cfg, "head_to_head_seed", 0))
            B = _cfg_int(plot_cfg, "head_to_head_bootstrap", 5000)
            if B > 0 and len(wv) > 1:
                draws = wv[rng.integers(0, len(wv), size=(B, len(wv)))].mean(axis=1)
                st["p_win_ci_low"] = float(np.percentile(draws, 2.5))
                st["p_win_ci_high"] = float(np.percentile(draws, 97.5))

    print(f"\n=== Head to head: {focus} vs {comp} (pre-specified) ===")
    print(f"  {st['n_transitions_focus_higher']}/{n_t} transitions and "
          f"{st['n_datasets_focus_higher']}/{n_d} datasets favour {focus}")
    print(f"  HL shift {st['hl_shift']:+.4f} "
          f"[{st['hl_ci_low']:+.4f}, {st['hl_ci_high']:+.4f}] (Fisher z)")
    print(f"  one-sided: signed-rank p={st['p_signrank_one_sided']:.4g}, "
          f"sign p={st['p_sign_one_sided']:.4g}, dataset sign-flip "
          f"p={st['p_signflip_one_sided']:.4g} (floor {st['signflip_floor']:.4g})")
    print(f"  direction: only {focus} correct on {st['only_focus_correct']} "
          f"transitions, only {comp} on {st['only_comparator_correct']}, "
          f"McNemar p={st['p_mcnemar']:.4g}")
    return per_edge, per_ds, st


def plot_head_to_head(per_edge, per_ds, st, plot_cfg, suffix):
    """Three views of one pre-specified pair: paired, per dataset, and by size."""
    if per_edge is None or per_edge.empty or not st:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    dpi = plot_cfg.get("dpi", 300)
    focus, comp = st["focus"], st["comparator"]
    palette = build_palette([focus, comp], plot_cfg.get("method_colors"))
    c_focus = palette.get(focus, "#2a78d6")
    c_comp = palette.get(comp, "#eb6834")
    ds_names = list(pd.unique(per_edge["dataset"]))
    marker_map = _dataset_marker_map(ds_names, plot_cfg)

    figsize = tuple(plot_cfg.get("head_to_head_figsize") or [15.5, 6.2])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    # --- paired scatter ----------------------------------------------------
    ax = axes[0]
    lim = float(np.nanmax(np.abs(per_edge[["focus", "comparator"]].to_numpy()))) * 1.12
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1.4, color="#111111", zorder=3)
    ax.fill_between([-lim, lim], [-lim, lim], [lim, lim], color=c_focus, alpha=0.06,
                    zorder=0)
    for d_name, g in per_edge.groupby("dataset"):
        ax.plot(g["comparator"], g["focus"], marker=marker_map.get(d_name, "o"),
                ms=7, linestyle="none", color=c_focus, markeredgecolor="#333333",
                markeredgewidth=0.45, alpha=0.9, zorder=4)
    ax.axhline(0, color="#777777", lw=1.0, ls=":", zorder=2)
    ax.axvline(0, color="#777777", lw=1.0, ls=":", zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # The quadrants ARE the McNemar table. Spelling them out here stops the
    # reversal counts from being read off the sorted-difference panel, which
    # answers a different question: which method scored higher, not which one
    # got the direction right.
    q = [(-lim * 0.97, lim * 0.97, "left", "top",
          f"only {focus} correct: {st['only_focus_correct']}"),
         (lim * 0.97, lim * 0.97, "right", "top",
          f"both correct: {st['both_correct']}"),
         (-lim * 0.97, -lim * 0.97, "left", "bottom",
          f"both reversed: {st['both_reversed']}"),
         (lim * 0.97, -lim * 0.97, "right", "bottom",
          f"only {comp} correct: {st['only_comparator_correct']}")]
    for qx, qy, ha, va, txt in q:
        ax.text(qx, qy, txt, ha=ha, va=va, fontsize=8, color="#52514e",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fcfcfb",
                          edgecolor="none", alpha=0.85), zorder=5)
    ax.set_xlabel(f"{comp}  ({VALUE_LABEL}, Fisher z)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"{focus}  ({VALUE_LABEL}, Fisher z)", fontsize=10.5, fontweight="bold")
    ax.set_title(f"Every transition, paired\n"
                 f"{st['n_transitions_focus_higher']}/{st['n_transitions']} above the "
                 f"line favour {focus}; quadrants = the direction table",
                 fontsize=11, fontweight="bold")

    # --- per dataset -------------------------------------------------------
    ax = axes[1]
    order_ds = list(per_ds.sort_values("diff")["dataset"])
    y = np.arange(len(order_ds))
    for yi, dn in zip(y, order_ds):
        r = per_ds[per_ds["dataset"] == dn].iloc[0]
        ax.plot([r["comparator"], r["focus"]], [yi, yi], color="#999999", lw=1.6,
                zorder=1)
        ax.plot([r["comparator"]], [yi], marker=marker_map.get(dn, "o"), ms=8,
                color=c_comp, markeredgecolor="#333333", markeredgewidth=0.5, zorder=3)
        ax.plot([r["focus"]], [yi], marker=marker_map.get(dn, "o"), ms=8,
                color=c_focus, markeredgecolor="#333333", markeredgewidth=0.5, zorder=3)
    draw_zero_line(ax, plot_cfg, vertical=False)
    ax.set_yticks(y)
    ax.set_yticklabels(order_ds)
    ax.set_xlabel(f"Median {VALUE_LABEL} per dataset (Fisher z)", fontsize=10.5,
                  fontweight="bold")
    ax.set_title(f"Dataset by dataset\n"
                 f"{st['n_datasets_focus_higher']}/{st['n_datasets']} favour {focus}",
                 fontsize=11, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=c_focus, linestyle="none",
                              markersize=7, markeredgecolor="#333333", label=focus),
                       Line2D([], [], marker="o", color=c_comp, linestyle="none",
                              markersize=7, markeredgecolor="#333333", label=comp)],
              fontsize=8, loc="best", frameon=True, framealpha=0.9)

    # --- paired differences, sorted ---------------------------------------
    ax = axes[2]
    srt = per_edge.sort_values("diff").reset_index(drop=True)
    cols = [c_focus if v > 0 else c_comp for v in srt["diff"]]
    ax.bar(np.arange(len(srt)), srt["diff"], color=cols, edgecolor="#333333",
           linewidth=0.4, width=0.85)
    draw_zero_line(ax, plot_cfg)
    ax.set_xticks([])
    ax.set_xlabel(f"{UNIT_LABEL_PLURAL}, sorted", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"{focus} − {comp}  (Fisher z)", fontsize=10.5, fontweight="bold")
    n_lo = st["n_transitions"] - st["n_transitions_focus_higher"]
    txt = (f"bars: {st['n_transitions_focus_higher']} favour {focus}, "
           f"{n_lo} favour {comp}\n"
           f"HL {st['hl_shift']:+.3f} [{st['hl_ci_low']:+.3f}, {st['hl_ci_high']:+.3f}]\n"
           f"signed-rank p = {st['p_signrank_one_sided']:.3g}\n"
           f"dataset sign-flip p = {st['p_signflip_one_sided']:.3g} "
           f"(floor {st['signflip_floor']:.3g})\n"
           f"direction (left panel, not these bars):\n"
           f"  only {focus} correct {st['only_focus_correct']}, "
           f"only {comp} {st['only_comparator_correct']}, "
           f"McNemar p = {st['p_mcnemar']:.3g}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color="#111111", bbox=dict(boxstyle="round,pad=0.4", facecolor="#fcfcfb",
                                       edgecolor="#c3c2b7", alpha=0.95))
    ax.set_title("Paired difference per transition\n"
                 "which method scored HIGHER — not who got the direction right",
                 fontsize=11, fontweight="bold")

    d_handles = [Line2D([], [], marker=marker_map[dn], color="#555555", linestyle="none",
                        markersize=6, markeredgecolor="#333333", markeredgewidth=0.4,
                        label=str(dn)) for dn in ds_names]
    ncol = _cfg_int(plot_cfg, "head_to_head_legend_ncol", 0) or min(len(d_handles), 7)
    fig.legend(handles=d_handles, title="Dataset", fontsize=8, title_fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False, ncol=ncol)
    fig.suptitle(plot_cfg.get("head_to_head_title")
                 or f"{focus} vs {comp}: a pre-specified head-to-head",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.09, 1, 0.92])
    _save_figure(f"{save_base}_head_to_head{suffix}", dpi)


def fit_cbdir_ceiling(long_df, method_order, plot_cfg):
    """Achieved CBDir against the ceiling the cone geometry allows.

    For a boundary cell, CBDir_i = <v_hat_i, m_i> with m_i the mean unit
    displacement to its target-cluster neighbours, so

        CBDir_i = Rbar_i * cos(theta_i),     Rbar_i = ||m_i||,

    and |CBDir_i| <= Rbar_i with equality when the velocity points along the
    resultant. Rbar is fixed by the kNN graph and the embedding: no method can
    move it. Only cos(theta_i) is the method's doing, so the fraction of the
    ceiling achieved, sum_i Rbar_i cos(theta_i) / sum_i Rbar_i, compares
    transitions whose cones differ in tightness — which raw CBDir cannot.

    The weighting is not incidental. A cell whose cone is nearly isotropic
    (Rbar ~ 0) carries no directional information and its per-cell ratio is
    numerically unstable; weighting by Rbar gives it the small say it deserves.

    Also reported: ||mean_i m_i||, the best a single shared direction could do.
    It is at most the mean of the per-cell ceilings, and the gap is the price a
    perfectly coherent field pays at that boundary — the formal counterpart of
    the CBDir-vs-ICCoh panel.
    """
    need = {"resultant_length", "cbdir"}
    if long_df is None or long_df.empty or not need.issubset(long_df.columns):
        return pd.DataFrame(), pd.DataFrame()
    d = long_df.dropna(subset=["cbdir", "resultant_length"])
    d = d[d["method"].isin(method_order)]
    if d.empty:
        print("  Skipping the ceiling analysis: no rows carry a resultant_length "
              "(the long tables predate it)")
        return pd.DataFrame(), pd.DataFrame()

    agg = {"cbdir": ("cbdir", "mean"), "ceiling_free": ("resultant_length", "mean"),
           "n_cells": ("cbdir", "size"),
           "sum_cb": ("cbdir", "sum"), "sum_r": ("resultant_length", "sum")}
    if "ceiling_coherent" in d.columns:
        agg["ceiling_coherent"] = ("ceiling_coherent", "first")
    if "n_target_neighbors" in d.columns:
        agg["median_n_target"] = ("n_target_neighbors", "median")
    pts = d.groupby(["dataset", "method", "edge"]).agg(**agg).reset_index()
    pts["alignment"] = np.where(pts["sum_r"] > 0, pts["sum_cb"] / pts["sum_r"], np.nan)
    if "ceiling_coherent" in pts.columns:
        pts["coherence_cost"] = pts["ceiling_free"] - pts["ceiling_coherent"]
    if "n_target_neighbors" in d.columns:
        # Rbar is inflated at small n: an isotropic cone of n neighbours still
        # reports about 1/sqrt(n). Correct only for cross-transition comparison;
        # the raw ceiling remains exact for the realised neighbour set.
        n = pts["median_n_target"].to_numpy(dtype=float)
        r = pts["ceiling_free"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rho2 = np.where(n > 1, (n * r ** 2 - 1.0) / (n - 1.0), np.nan)
        pts["ceiling_free_corrected"] = np.sqrt(np.clip(rho2, 0.0, 1.0))

    breach = pts[pts["cbdir"].abs() > pts["ceiling_free"] + 1e-9]
    if not breach.empty:
        print(f"  WARNING: {len(breach)} transition(s) report |CBDir| above their own "
              f"ceiling — impossible by construction; check that the ceiling and the "
              f"metric were computed in the same space and truncation")

    # The cone geometry belongs to the graph and the embedding, not to a method.
    # If it moves between methods, their h5ads carry different PCAs or graphs and
    # the CBDir comparison is not like for like.
    spread = (pts.groupby(["dataset", "edge"])["ceiling_free"]
                 .agg(lambda s: float(s.max() - s.min())))
    worst = float(spread.max()) if len(spread) else np.nan

    rng = np.random.default_rng(_cfg_int(plot_cfg, "ceiling_seed", 0))
    B = _cfg_int(plot_cfg, "ceiling_bootstrap", 5000)
    how = str(plot_cfg.get("ceiling_aggregator", "median") or "median").lower()
    agg = (lambda s: float(np.nanmedian(s))) if how.startswith("med") \
        else (lambda s: float(np.nanmean(s)))
    ds_names = list(pd.unique(pts["dataset"]))
    rows = []
    for m in method_order:
        sub = pts[pts["method"] == m]
        if sub.empty:
            continue
        # The unit of replication is the TRANSITION, nested in the DATASET — as
        # everywhere else in this pipeline. Pooling cells instead would let one
        # transition with thousands of cells outvote twenty small ones; on real
        # data that inversion is large enough to reorder the methods, so the
        # cell-weighted figure is kept only as a clearly named diagnostic.
        per_ds = sub.groupby("dataset")["alignment"].apply(agg)
        avail = [k for k in ds_names if k in per_ds.index]
        lo = hi = np.nan
        if B > 0 and len(avail) > 1:
            vals = per_ds.loc[avail].to_numpy(dtype=float)
            draws = np.array([float(np.nanmean(vals[rng.integers(0, len(avail), len(avail))]))
                              for _ in range(B)])
            lo, hi = float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))
        denom = float(sub["sum_r"].sum())
        rows.append(dict(
            method=m, n_transitions=int(len(sub)), n_datasets=int(sub["dataset"].nunique()),
            mean_cbdir=float(sub["cbdir"].mean()),
            mean_ceiling_free=float(sub["ceiling_free"].mean()),
            mean_ceiling_coherent=(float(sub["ceiling_coherent"].mean())
                                   if "ceiling_coherent" in sub.columns else np.nan),
            alignment=float(np.nanmean(per_ds.to_numpy(dtype=float))),
            alignment_ci_low=lo, alignment_ci_high=hi,
            alignment_transition_median=float(np.nanmedian(sub["alignment"])),
            alignment_transition_mean=float(np.nanmean(sub["alignment"])),
            alignment_cell_weighted=(float(sub["sum_cb"].sum() / denom)
                                     if denom > 0 else np.nan),
            alignment_dataset_median=float(np.nanmedian(per_ds.to_numpy(dtype=float))),
            alignment_worst_dataset=float(np.nanmin(per_ds.to_numpy(dtype=float))),
            n_datasets_above_zero=int((per_ds.to_numpy(dtype=float) > 0).sum()),
            aggregator=("median" if how.startswith("med") else "mean"),
            n_transitions_at_half_ceiling=int((sub["alignment"] > 0.5).sum()),
            ceiling_spread_across_methods=worst))
    summary = pd.DataFrame(rows).sort_values("alignment", ascending=False)

    print("\n=== Achieved vs attainable (CBDir = Rbar x cos theta) ===")
    print(f"    alignment = fraction of the geometric ceiling reached, "
          f"Rbar-weighted within a transition, then {how} over a dataset's "
          f"transitions and averaged over datasets")
    if np.isfinite(worst):
        tol = _cfg_num(plot_cfg, "ceiling_spread_warn", 0.02)
        note = ("the cone geometry is shared, so CBDir is comparable across methods"
                if worst < tol else
                "the ceiling MOVES between methods, so their embeddings or graphs "
                "differ — CBDir is not strictly like for like")
        print(f"    max ceiling spread across methods within a transition: {worst:.4f} — {note}")
    for _, r in summary.iterrows():
        print(f"  {r['method']:<20s} CBDir={r['mean_cbdir']:+.4f}  "
              f"ceiling={r['mean_ceiling_free']:.4f}  "
              f"alignment={r['alignment']:+.4f} "
              f"[{r['alignment_ci_low']:+.4f}, {r['alignment_ci_high']:+.4f}]  "
              f"| per-transition median {r['alignment_transition_median']:+.4f}, "
              f"worst dataset {r['alignment_worst_dataset']:+.4f}, "
              f"{int(r['n_datasets_above_zero'])}/{int(r['n_datasets'])} datasets above zero")
    print("    the headline averages the per-dataset values, so a method with a high "
          "median but one bad dataset ranks below a consistent one — the two columns "
          "can disagree, and both are printed")
    return pts, summary


def plot_cbdir_ceiling(pts, summary, method_order, plot_cfg, suffix):
    """Three panels: inside the wedge, how much of the ceiling, what coherence costs."""
    if pts is None or pts.empty or summary is None or summary.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    order = [m for m in summary["method"] if m in set(pts["method"])]
    by_shape = bool(plot_cfg.get("ceiling_dataset_markers", True))
    ds_names = list(pd.unique(pts["dataset"]))
    marker_map = _dataset_marker_map(ds_names, plot_cfg) if by_shape else {}
    has_coh = "ceiling_coherent" in pts.columns and pts["ceiling_coherent"].notna().any()

    n_panels = 3 if has_coh else 2
    figsize = tuple(plot_cfg.get("ceiling_figsize")
                    or [5.6 * n_panels, 0.30 * max(len(order), 3) + 4.4])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, dpi=dpi)

    # --- 1: achieved against attainable -----------------------------------
    ax = axes[0]
    hi = float(max(pts["ceiling_free"].max(), pts["cbdir"].abs().max())) * 1.05
    ax.plot([0, hi], [0, hi], ls="--", lw=1.4, color="#111111", zorder=4)
    ax.plot([0, hi], [0, -hi], ls="--", lw=1.4, color="#111111", zorder=4, alpha=0.5)
    ax.fill_between([0, hi], [0, hi], [0, -hi], color="#2a78d6", alpha=0.05, zorder=0)
    for m in order:
        sub = pts[pts["method"] == m]
        col = palette.get(m, "#888888")
        if by_shape:
            for d_name, g in sub.groupby("dataset"):
                ax.plot(g["ceiling_free"], g["cbdir"], marker=marker_map.get(d_name, "o"),
                        ms=6.5, alpha=0.8, color=col, markeredgecolor="#333333",
                        markeredgewidth=0.4, linestyle="none")
        else:
            ax.plot(sub["ceiling_free"], sub["cbdir"], "o", ms=5, alpha=0.75, color=col,
                    markeredgecolor="#333333", markeredgewidth=0.3, linestyle="none")
    draw_zero_line(ax, plot_cfg)
    ax.set_xlim(0, hi)
    ax.set_xlabel("Ceiling: mean resultant length $\\bar{R}$ of the target cone",
                  fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"{VALUE_LABEL} achieved", fontsize=10.5, fontweight="bold")
    ax.set_title("Every transition sits inside the wedge\n"
                 "the dashed lines are $\\pm\\bar{R}$ — geometry, not performance",
                 fontsize=10.5, fontweight="bold")

    # --- 2: fraction of the ceiling ---------------------------------------
    ax = axes[1]
    rows = order[::-1]
    y = np.arange(len(rows))
    how_s = (str(summary["aggregator"].iloc[0]) if "aggregator" in summary.columns
             else "median")
    # The per-dataset values the headline averages, drawn behind it. Without them
    # the ranking looks wrong to anyone reading medians off the boxplot: a method
    # with a high median and one bad dataset sits BELOW a consistent one here, and
    # the dots are what make that legible rather than surprising.
    per_ds_pts = (pts.groupby(["method", "dataset"])["alignment"]
                     .agg("median" if how_s.startswith("med") else "mean")
                     .reset_index())
    for yi, m in zip(y, rows):
        g = per_ds_pts[per_ds_pts["method"] == m]
        col = palette.get(m, "#888888")
        for _, rr in g.iterrows():
            ax.plot([rr["alignment"]], [yi + 0.0], marker=marker_map.get(rr["dataset"], "o")
                    if by_shape else "o", ms=5.5, linestyle="none", color=col,
                    alpha=0.45, markeredgecolor="#333333", markeredgewidth=0.3, zorder=2)
        r = summary[summary["method"] == m].iloc[0]
        if np.isfinite(r["alignment_ci_low"]) and np.isfinite(r["alignment_ci_high"]):
            ax.hlines(yi, r["alignment_ci_low"], r["alignment_ci_high"],
                      color="#52514e", lw=2.0, zorder=3)
        ax.plot([r["alignment"]], [yi], "o", ms=9, color=col,
                markeredgecolor="#111111", markeredgewidth=0.8, zorder=4)
    draw_zero_line(ax, plot_cfg, vertical=False)
    ax.axvline(1.0, ls=":", lw=1.2, color="#111111")
    ax.text(0.995, -0.42, "ceiling ", fontsize=8, color="#52514e",
            va="bottom", ha="right", rotation=90)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_xlabel("Alignment  $\\sum \\bar{R}\\cos\\theta / \\sum \\bar{R}$",
                  fontsize=10.5, fontweight="bold")
    ax.set_title(f"Fraction of the attainable direction captured\n"
                 f"filled dot = mean over datasets of the within-dataset {how_s}\n"
                 f"faint dots = the {len(pd.unique(per_ds_pts['dataset']))} datasets "
                 f"it averages; 95% CI from a cluster bootstrap",
                 fontsize=10.5, fontweight="bold")

    # --- 3: what a coherent field would cost ------------------------------
    if has_coh:
        ax = axes[2]
        geo = (pts.groupby(["dataset", "edge"])
                  .agg(free=("ceiling_free", "mean"), coh=("ceiling_coherent", "mean"))
                  .reset_index())
        lim = float(max(geo["free"].max(), geo["coh"].max())) * 1.05
        ax.plot([0, lim], [0, lim], ls="--", lw=1.4, color="#111111", zorder=3)
        if by_shape:
            for d_name, g in geo.groupby("dataset"):
                ax.plot(g["free"], g["coh"], marker=marker_map.get(d_name, "o"), ms=6.5,
                        color="#52514e", alpha=0.75, markeredgecolor="#333333",
                        markeredgewidth=0.4, linestyle="none")
        else:
            ax.plot(geo["free"], geo["coh"], "o", ms=5, color="#52514e", alpha=0.75,
                    markeredgecolor="#333333", markeredgewidth=0.3, linestyle="none")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Free-field ceiling  mean $\\bar{R}$", fontsize=10.5,
                      fontweight="bold")
        ax.set_ylabel("Coherent-field ceiling  $\\|$mean $m_i\\|$", fontsize=10.5,
                      fontweight="bold")
        ax.set_title("What one shared direction costs\n"
                     "distance below the line = the price of a perfectly smooth field",
                     fontsize=10.5, fontweight="bold")

    from matplotlib.lines import Line2D
    m_handles = [Line2D([], [], marker="o", color=palette.get(m, "#888888"),
                        linestyle="none", markersize=6, markeredgecolor="#333333",
                        markeredgewidth=0.4, label=m) for m in order]
    ncol_m = _cfg_int(plot_cfg, "ceiling_legend_ncol", 0) or min(len(m_handles), 5)
    leg_rows = int(np.ceil(len(m_handles) / ncol_m))
    leg_m = fig.legend(handles=m_handles, title="Method", fontsize=8, title_fontsize=9,
                       loc="lower center", bbox_to_anchor=(0.28, 0.01), frameon=False,
                       ncol=ncol_m)
    fig.add_artist(leg_m)
    if by_shape:
        d_handles = [Line2D([], [], marker=marker_map[dn], color="#555555",
                            linestyle="none", markersize=6, markeredgecolor="#333333",
                            markeredgewidth=0.4, label=str(dn)) for dn in ds_names]
        ncol_d = (_cfg_int(plot_cfg, "ceiling_dataset_legend_ncol", 0)
                  or min(len(d_handles), 4))
        leg_rows = max(leg_rows, int(np.ceil(len(d_handles) / ncol_d)))
        fig.legend(handles=d_handles, title="Dataset", fontsize=8, title_fontsize=9,
                   loc="lower center", bbox_to_anchor=(0.78, 0.01), frameon=False,
                   ncol=ncol_d)

    fig.suptitle(plot_cfg.get("ceiling_title",
                              "How much of the attainable direction does each method get?"),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03 + 0.042 * (leg_rows + 0.5), 1, 0.94])
    _save_figure(f"{save_base}_ceiling{suffix}", dpi)


_ICCOH_FIXED_LABELS = {
    "iccoh_embedding": "ICCoh (embedding, velocity_pca)",
    "iccoh_gene_raw": "ICCoh (gene space, raw cosine)",
    "iccoh_gene_vst": "ICCoh (gene space, VST raw cosine)",
}


def iccoh_provenance(long_df, column="iccoh"):
    """Which ICCoh a column holds, and whether every loaded table agrees.

    compute_cbdir_run.py records iccoh_space / iccoh_version /
    iccoh_neighbor_graph / iccoh_gene_set per row. This reads them back so the
    figures say which ICCoh they show, and so a run that silently mixes versions
    — one dataset recomputed with other settings, or methods run with different
    intersection modes — is caught instead of being fitted as one covariate.

    Returns (label, per-(dataset, method) provenance table, mixed).
    """
    if column in _ICCOH_FIXED_LABELS:
        return _ICCOH_FIXED_LABELS[column], pd.DataFrame(), False
    if long_df is None or "iccoh_space" not in long_df.columns:
        return ("ICCoh (embedding; the long tables predate the provenance columns)",
                pd.DataFrame(), False)
    d = long_df[long_df[column].notna()] if column in long_df.columns else long_df
    keys = [c for c in ("iccoh_space", "iccoh_version", "iccoh_neighbor_graph",
                        "iccoh_gene_set") if c in d.columns]
    prov = d[["dataset", "method"] + keys].copy()
    for c in keys:
        prov[c] = prov[c].fillna("").astype(str)
    if column in ("iccoh_owngenes", "iccoh_sharedgenes"):
        prov["iccoh_gene_set"] = "own" if column == "iccoh_owngenes" else "shared"
        keys = list(dict.fromkeys(keys + ["iccoh_gene_set"]))
    prov = prov.drop_duplicates().reset_index(drop=True)
    combos = prov[keys].drop_duplicates()
    if combos.empty:
        return "ICCoh", prov, False
    r = combos.iloc[0]
    space = r.get("iccoh_space", "")
    if space == "gene_confidence":
        gs = r.get("iccoh_gene_set", "")
        label = (f"ICCoh (gene space, {r.get('iccoh_version', '')}, "
                 f"{r.get('iccoh_neighbor_graph', '')}"
                 + (f", {gs} genes" if gs else "") + ")")
    elif space == "gene":
        label = "ICCoh (gene space, raw cosine)"
    else:
        label = "ICCoh (embedding, velocity_pca)"
    return label, prov, len(combos) > 1


def fit_cbdir_vs_iccoh(long_df, method_order, plot_cfg):
    """CBDir against in-cluster coherence, and CBDir at matched coherence.

    The arrow CBDir scores is a transition-weighted average of displacements to
    the very neighbours it is then scored against, so a method whose velocity
    field is locally smooth gets boundary correctness partly for free. ICCoh
    measures that smoothness directly and is blind to orientation, which makes it
    the right covariate rather than a rival metric: the question is not which
    method is most coherent, it is whether a method's CBDir survives being read
    at matched coherence.

    The fit carries a per-dataset intercept, so the trend is estimated WITHIN
    datasets. Pooling across datasets would let a dataset that is both easy and
    coherent masquerade as evidence for the trend, which is the confound the
    panel exists to rule out. The residual is therefore "CBDir above what this
    dataset's coherence-to-correctness exchange rate would predict".
    """
    col = str(plot_cfg.get("iccoh_column", "iccoh") or "iccoh")
    if long_df is None or long_df.empty or col not in long_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    label, prov, mixed = iccoh_provenance(long_df, col)
    if mixed:
        print(f"\n  WARNING: the long tables carry more than one kind of '{col}' "
              f"(space / version / graph / gene set differ across datasets or methods):")
        print("  " + prov.to_string(index=False).replace("\n", "\n  "))
        if not plot_cfg.get("iccoh_allow_mixed", False):
            print("  Skipping CBDir-vs-ICCoh: a covariate that means different things "
                  "for different points is not one covariate. Recompute with one "
                  "setting, or set iccoh_allow_mixed: true to fit anyway.")
            return pd.DataFrame(), pd.DataFrame()
    if col != "iccoh":
        long_df = long_df.assign(iccoh=long_df[col])
    transform = str(plot_cfg.get("transform", "atanh")).lower()
    eps = _cfg_num(plot_cfg, "transform_eps", 1e-6)
    d = long_df.dropna(subset=["cbdir", "iccoh"])
    if d.empty:
        print("  Skipping CBDir-vs-ICCoh: no rows carry an iccoh value "
              "(compute_iccoh was off, or the long tables predate it)")
        return pd.DataFrame(), pd.DataFrame()

    z, _ = _forward_transform(d["cbdir"].to_numpy(), transform, eps)
    d = d.assign(_z=z)
    if "source" not in d.columns:
        d = d.assign(source=d["edge"].astype(str).str.split(" -> ").str[0])
    unit = str(plot_cfg.get("iccoh_point_unit", "dataset")).lower()
    how = str(plot_cfg.get("iccoh_aggregator", "median")).lower()
    agg = (lambda s: float(np.nanmedian(s))) if how.startswith("med") \
        else (lambda s: float(np.nanmean(s)))

    # Two-stage, and the stages are not interchangeable. CBDir is a per-cell
    # score within a TRANSITION; ICCoh is a per-cell score within a CLUSTER, and
    # a source cell is re-scored once per edge leaving its cluster. Collapsing
    # cells first — CBDir by transition, ICCoh by cluster on each cell counted
    # once — keeps a fork like Pre-endocrine -> {Alpha, Beta, Delta, Epsilon}
    # from entering the coherence side four times.
    per_edge = (d.groupby(["dataset", "method", "edge"])
                  .agg(cbdir_z=("_z", agg), n_cells=("_z", "size"))
                  .reset_index())
    cells = d.drop_duplicates(["dataset", "method", "source", "cell_barcode"]) \
        if "cell_barcode" in d.columns else d
    per_cluster = (cells.groupby(["dataset", "method", "source"])
                        .agg(iccoh=("iccoh", agg))
                        .reset_index())

    if unit.startswith("d"):
        # One point per (dataset, method): the dataset's transitions and its
        # source clusters each collapsed again, so no dataset speaks louder for
        # having been cut into more pieces.
        a = (per_edge.groupby(["dataset", "method"])
                     .agg(cbdir_z=("cbdir_z", agg), n_cells=("n_cells", "sum"),
                          n_transitions=("edge", "size")).reset_index())
        b = (per_cluster.groupby(["dataset", "method"])
                        .agg(iccoh=("iccoh", agg), n_clusters=("source", "size"))
                        .reset_index())
        pts = a.merge(b, on=["dataset", "method"], how="inner")
    else:
        src = d.drop_duplicates(["dataset", "method", "edge"])[
            ["dataset", "method", "edge", "source"]]
        pts = (per_edge.merge(src, on=["dataset", "method", "edge"], how="left")
                       .merge(per_cluster, on=["dataset", "method", "source"], how="left"))
    pts = pts[pts["method"].isin(method_order)].dropna(subset=["cbdir_z", "iccoh"])
    pts["point_unit"] = "dataset" if unit.startswith("d") else "transition"
    pts["aggregator"] = "median" if how.startswith("med") else "mean"
    pts["iccoh_label"] = label
    pts["iccoh_column"] = col
    if pts["dataset"].nunique() < 1 or len(pts) < 4:
        return pts, pd.DataFrame()

    # With dataset intercepts in the model, the slope is identified only by the
    # WITHIN-dataset spread of ICCoh. If that is (near) zero — one cluster per
    # dataset, or a constant coherence — the design is collinear and OLS returns
    # an arbitrary slope with an enormous SE rather than failing. Stop instead.
    within = pts["iccoh"] - pts.groupby("dataset")["iccoh"].transform("mean")
    if float(np.nanstd(within.to_numpy(dtype=float))) < 1e-9:
        print("  Skipping CBDir-vs-ICCoh: ICCoh has no within-dataset variation, "
              "so the slope is not identified alongside the dataset intercepts")
        return pts, pd.DataFrame()

    # CBDir ~ ICCoh + dataset (fixed effects), fitted on all methods at once:
    # one exchange rate per panel, so the per-method residuals are comparable.
    X = pd.get_dummies(pts["dataset"], prefix="ds", drop_first=False).astype(float)
    X.insert(0, "iccoh", pts["iccoh"].to_numpy(dtype=float))
    y = pts["cbdir_z"].to_numpy(dtype=float)
    try:
        import statsmodels.api as sm
        fit = sm.OLS(y, X.to_numpy(dtype=float)).fit()
        slope = float(fit.params[0])
        slope_p = float(fit.pvalues[0])
        slope_se = float(fit.bse[0])
        pts["fitted"] = fit.fittedvalues
        pts["residual"] = y - fit.fittedvalues
        r2 = float(fit.rsquared)
    except Exception as e:
        print(f"  WARNING: CBDir-vs-ICCoh fit failed: {type(e).__name__}: {e}")
        return pts, pd.DataFrame()

    rng = np.random.default_rng(_cfg_int(plot_cfg, "iccoh_seed", 0))
    B = _cfg_int(plot_cfg, "iccoh_bootstrap", 5000)
    ds_names = list(pd.unique(pts["dataset"]))
    rows = []
    for m in method_order:
        sub = pts[pts["method"] == m]
        if sub.empty:
            continue
        # Cluster bootstrap over datasets: the residuals of one dataset's
        # transitions share its cells, its annotation and its intercept.
        by_ds = {k: g["residual"].to_numpy(dtype=float) for k, g in sub.groupby("dataset")}
        avail = [k for k in ds_names if k in by_ds]
        lo = hi = np.nan
        if B > 0 and len(avail) > 1:
            draws = np.array([
                np.mean(np.concatenate([by_ds[avail[i]]
                                        for i in rng.integers(0, len(avail), len(avail))]))
                for _ in range(B)])
            lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        rows.append(dict(
            method=m, n_points=int(len(sub)), n_datasets=int(sub["dataset"].nunique()),
            mean_iccoh=float(sub["iccoh"].mean()), mean_cbdir_z=float(sub["cbdir_z"].mean()),
            mean_residual=float(sub["residual"].mean()),
            residual_ci_low=lo, residual_ci_high=hi,
            slope_iccoh=slope, slope_se=slope_se, slope_p=slope_p, model_r2=r2,
            point_unit=("dataset" if unit.startswith("d") else "transition"),
            aggregator=("median" if how.startswith("med") else "mean"),
            iccoh_label=label, iccoh_column=col))
    summary = pd.DataFrame(rows).sort_values("mean_residual", ascending=False)

    print("\n=== CBDir vs ICCoh (coherence control) ===")
    print(f"    covariate: column '{col}' = {label}")
    print(f"    {len(pts)} points, one per "
          f"({'dataset, method' if unit.startswith('d') else 'dataset, transition, method'}), "
          f"aggregated by {'median' if how.startswith('med') else 'mean'}")
    print(f"    CBDir_z ~ ICCoh + dataset:  slope = {slope:+.4f} "
          f"(SE {slope_se:.4f}, p = {slope_p:.3g}), R2 = {r2:.3f}")
    print("    residual = CBDir above what this dataset's coherence would predict; "
          "ICCoh is orientation-blind and is never itself a correctness score")
    for _, r in summary.iterrows():
        print(f"  {r['method']:<20s} ICCoh={r['mean_iccoh']:+.4f}  "
              f"CBDir(z)={r['mean_cbdir_z']:+.4f}  "
              f"residual={r['mean_residual']:+.4f} "
              f"[{r['residual_ci_low']:+.4f}, {r['residual_ci_high']:+.4f}]")
    return pts, summary


def plot_cbdir_vs_iccoh(pts, summary, method_order, plot_cfg, suffix):
    """Left: CBDir against ICCoh with the within-dataset trend. Right: residuals.

    A method sitting above the line has boundary correctness its coherence does
    not explain — the claim the panel is built to support or refute. A method on
    the line may still be perfectly good; it just cannot claim the correctness is
    independent of the smoothness of its field.
    """
    if pts is None or pts.empty or summary is None or summary.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    order = [m for m in summary["method"] if m in set(pts["method"])]

    figsize = tuple(plot_cfg.get("iccoh_figsize")
                    or [13.0, 0.34 * max(len(order), 3) + 4.2])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    by_shape = bool(plot_cfg.get("iccoh_dataset_markers", True))
    ds_names = list(pd.unique(pts["dataset"]))
    marker_map = _dataset_marker_map(ds_names, plot_cfg) if by_shape else {}
    ms = _cfg_num(plot_cfg, "iccoh_marker_size", 7.5 if by_shape else 5)

    ax = axes[0]
    for m in order:
        sub = pts[pts["method"] == m]
        col = palette.get(m, "#888888")
        if by_shape:
            # Colour carries the method, shape carries the dataset — the same
            # pairing the stability panels use, so a reader moving between the
            # two figures does not have to relearn the encoding.
            for d_name, g in sub.groupby("dataset"):
                ax.plot(g["iccoh"], g["cbdir_z"], marker=marker_map.get(d_name, "o"),
                        ms=ms, alpha=0.8, color=col, markeredgecolor="#333333",
                        markeredgewidth=0.4, linestyle="none")
        else:
            ax.plot(sub["iccoh"], sub["cbdir_z"], "o", ms=ms, alpha=0.75,
                    color=col, markeredgecolor="#333333",
                    markeredgewidth=0.3, linestyle="none")
    slope = float(summary["slope_iccoh"].iloc[0])
    # The fitted line is drawn at the mean dataset intercept: the panel shows one
    # exchange rate, while the fit itself kept every dataset's own level.
    xs = np.linspace(float(pts["iccoh"].min()), float(pts["iccoh"].max()), 50)
    intercept = float(pts["cbdir_z"].mean() - slope * pts["iccoh"].mean())
    trend = ax.plot(xs, intercept + slope * xs, ls="--", lw=1.6, color="#111111",
                    zorder=4,
                    label="within-dataset trend")[0]
    ax.text(0.015, 0.985,
            f"within-dataset slope {slope:+.3f}  (p = {float(summary['slope_p'].iloc[0]):.3g})",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color="#111111", fontweight="bold")
    draw_zero_line(ax, plot_cfg)
    x_lab = (str(pts["iccoh_label"].iloc[0]) if "iccoh_label" in pts.columns
             else "In-cluster coherence (ICCoh)")
    ax.set_xlabel(f"{x_lab} of the source cells", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{VALUE_LABEL} (Fisher z)", fontsize=11, fontweight="bold")
    p_unit = str(pts["point_unit"].iloc[0]) if "point_unit" in pts.columns else "transition"
    p_agg = str(pts["aggregator"].iloc[0]) if "aggregator" in pts.columns else "mean"
    ax.set_title("Correctness against coherence\n"
                 "ICCoh is orientation-blind: it is the covariate, not a score\n"
                 + (f"one point per dataset and method ({p_agg} of {p_agg}s)"
                    if p_unit == "dataset" else
                    f"one point per transition and method ({p_agg} over cells)"),
                 fontsize=11, fontweight="bold")

    # Both keys go under the figure: inside the axes they land on the data, and
    # the point of the panel is the cloud, not the box covering it.
    from matplotlib.lines import Line2D
    m_handles = [Line2D([], [], marker="o", color=palette.get(m, "#888888"),
                        linestyle="none", markersize=6, markeredgecolor="#333333",
                        markeredgewidth=0.4, label=m) for m in order]
    n_m = len(m_handles) + 1
    ncol_m = _cfg_int(plot_cfg, "iccoh_legend_ncol", 0) or min(n_m, 5)
    leg_rows = int(np.ceil(n_m / ncol_m))
    leg_m = fig.legend(handles=m_handles + [trend], title="Method", fontsize=8,
                       title_fontsize=9, loc="lower center",
                       bbox_to_anchor=(0.28, 0.01), frameon=False, ncol=ncol_m)
    fig.add_artist(leg_m)
    if by_shape:
        d_handles = [Line2D([], [], marker=marker_map[dn], color="#555555",
                            linestyle="none", markersize=6, markeredgecolor="#333333",
                            markeredgewidth=0.4, label=str(dn)) for dn in ds_names]
        # Under the right panel, not beside the method key: the two keys read as
        # one run-on line when they share a row.
        ncol_d = _cfg_int(plot_cfg, "iccoh_dataset_legend_ncol", 0) or min(len(d_handles), 4)
        leg_rows = max(leg_rows, int(np.ceil(len(d_handles) / ncol_d)))
        fig.legend(handles=d_handles, title="Dataset", fontsize=8, title_fontsize=9,
                   loc="lower center", bbox_to_anchor=(0.80, 0.01), frameon=False,
                   ncol=ncol_d)

    ax = axes[1]
    rows = order[::-1]
    y = np.arange(len(rows))
    for yi, m in zip(y, rows):
        r = summary[summary["method"] == m].iloc[0]
        col = palette.get(m, "#888888")
        if np.isfinite(r["residual_ci_low"]) and np.isfinite(r["residual_ci_high"]):
            ax.hlines(yi, r["residual_ci_low"], r["residual_ci_high"], color="#52514e", lw=2.0)
        ax.plot([r["mean_residual"]], [yi], "o", ms=9, color=col,
                markeredgecolor="#111111", markeredgewidth=0.8, zorder=3)
    draw_zero_line(ax, plot_cfg, vertical=False)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_xlabel(f"{VALUE_LABEL} at matched coherence (residual, Fisher z)",
                  fontsize=11, fontweight="bold")
    ax.set_title("Above zero: correctness the coherence does not explain\n"
                 "95% CI from a dataset cluster bootstrap",
                 fontsize=11, fontweight="bold")

    fig.suptitle(plot_cfg.get("iccoh_title",
                              "Is boundary correctness bought by coherence?"),
                 fontsize=13, fontweight="bold")
    # Reserve the strip the two keys sit in, so they never overlap the axes.
    bottom = 0.03 + 0.042 * (leg_rows + 0.5)
    fig.tight_layout(rect=[0, bottom, 1, 0.94])
    _save_figure(f"{save_base}_cbdir_vs_iccoh{suffix}", dpi)


# ---------------------------------------------------------------------------
# Gene-count bias of gene-space ICCoh
# ---------------------------------------------------------------------------

def load_iccoh_gene_tables(dataset_dir_paths, dataset_order, str_suffix=None):
    """Read the per-dataset _iccoh_genes / _iccoh_gene_clusters side tables."""
    suffix = _suffix_str(str_suffix)
    genes, clus = [], []
    for d in dataset_order:
        base = dataset_dir_paths.get(d)
        if not base:
            continue
        for lst, tag in ((genes, "iccoh_genes"), (clus, "iccoh_gene_clusters")):
            pth = os.path.join(base, f"{d}_{tag}{suffix}.csv")
            if os.path.exists(pth):
                lst.append(pd.read_csv(pth))
                print(f"  Loaded {pth}")
    return (pd.concat(genes, ignore_index=True) if genes else pd.DataFrame(),
            pd.concat(clus, ignore_index=True) if clus else pd.DataFrame())


def _genes_from_long(long_df):
    """Fallback gene table from the long table's per-row gene columns."""
    need = {"iccoh_n_genes_own", "iccoh_n_genes_shared"}
    if long_df is None or not need.issubset(long_df.columns):
        return pd.DataFrame()
    cols = [c for c in ("iccoh_n_genes_own", "iccoh_n_genes_shared", "iccoh_pr_own",
                        "iccoh_eff_genes_own", "iccoh_pr_shared", "iccoh_eff_genes_shared")
            if c in long_df.columns]
    g = long_df.groupby(["dataset", "method"])[cols].first().reset_index()
    g.columns = [c.replace("iccoh_", "") for c in g.columns]
    return g.dropna(subset=["n_genes_own"])


def _gene_clusters_from_long(long_df):
    """Fallback per-cluster table (own / shared, full sets only) from the long
    table, each source cell counted once. Used when the side table is absent."""
    if long_df is None or "iccoh_owngenes" not in long_df.columns:
        return pd.DataFrame()
    d = long_df.copy()
    if "source" not in d.columns:
        d["source"] = d["edge"].astype(str).str.split(" -> ").str[0]
    d = d.drop_duplicates(["dataset", "method", "source", "cell_barcode"])
    rows = []
    for gs, col, ncol in (("own", "iccoh_owngenes", "iccoh_n_genes_own"),
                          ("shared", "iccoh_sharedgenes", "iccoh_n_genes_shared")):
        if col not in d.columns or not np.isfinite(d[col].to_numpy(float)).any():
            continue
        g = (d.groupby(["dataset", "method", "source"])
               .agg(median_iccoh=(col, "median"), mean_iccoh=(col, "mean"),
                    n_cells=(col, lambda s: int(np.isfinite(s).sum())),
                    k=(ncol, "first")).reset_index())
        g["gene_set"], g["kind"], g["rep"] = gs, "full", 0
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collapse_clusters(df, by, how):
    """Per-cluster summaries -> one value per `by` group (median of medians or
    mean of means), each cluster counted once, as the coherence control does."""
    col = "median_iccoh" if how == "median" else "mean_iccoh"
    f = np.nanmedian if how == "median" else np.nanmean

    def _one(s):
        v = s.to_numpy(dtype=float)
        return float(f(v)) if np.isfinite(v).any() else np.nan
    return df.groupby(by)[col].agg(_one).reset_index(name="iccoh")


def _fe_slope(y, x, fe, min_within_sd=0.05):
    """OLS slope of y on x with fixed effects for each label array in `fe`.

    Returns (slope, se, p, n) or None when x has too little variation left after
    the fixed effects: the slope is then not identified, and OLS would return an
    arbitrary number rather than failing.
    """
    import statsmodels.api as sm
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    parts = []
    for i, lab in enumerate(fe):
        dm = pd.get_dummies(pd.Series(lab).astype(str), drop_first=(i > 0)).astype(float)
        parts.append(dm.to_numpy())
    Z = np.column_stack(parts) if parts else np.ones((len(y), 1))
    beta_z = np.linalg.lstsq(Z, x, rcond=None)[0]
    # x is log2(gene count): less than `min_within_sd` doublings of spread left
    # after the fixed effects (e.g. counts that are dataset x method up to
    # rounding) cannot support a per-doubling slope, however small its SE looks
    if np.std(x - Z @ beta_z) < min_within_sd or len(y) <= Z.shape[1] + 1:
        return None
    fit = sm.OLS(y, np.column_stack([x, Z])).fit()
    return float(fit.params[0]), float(fit.bse[0]), float(fit.pvalues[0]), int(len(y))


def _cluster_boot(frame, stat, B, rng, group="dataset"):
    """Percentile CI of stat(frame) from resampling whole datasets."""
    if B <= 0:
        return np.nan, np.nan
    groups = list(pd.unique(frame[group]))
    if len(groups) < 2:
        return np.nan, np.nan
    parts = {g: frame[frame[group] == g] for g in groups}
    draws = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        f = pd.concat([parts[groups[i]].assign(**{group: f"{groups[i]}#{j}"})
                       for j, i in enumerate(pick)], ignore_index=True)
        v = stat(f)
        if v is not None and np.isfinite(v):
            draws.append(v)
    if len(draws) < max(20, B // 10):
        return np.nan, np.nan
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fit_iccoh_gene_bias(long_df, method_order, plot_cfg, genes=None, gene_clusters=None):
    """Does the number of genes a method is scored on bias its gene-space ICCoh?

    Three analyses, from weakest to strongest causal footing. The unit is the
    (dataset, method) pair throughout; clusters are collapsed first.

    A. Across methods (observational). ICCoh on each method's own genes against
       log2(own gene count):
         A1  + dataset fixed effects       - between methods within a dataset;
                                             confounded with method quality.
         A2  + dataset AND method effects  - within a method, across datasets:
                                             does a method score higher where it
                                             happened to fit more genes?
    B. Own vs shared genes (paired, same velocities). delta = ICCoh(own) -
       ICCoh(shared) against log2(n_own / n_shared), fitted through the origin
       (identical sets give delta = 0 by construction, so those pairs are shown
       but carry no information and are left out of the fit). This is what
       turning iccoh_intersection_mode off does to a method: gene COUNT and gene
       COMPOSITION change together.
    C. Dose-response (causal for count alone). Random subsets of each method's
       own genes: slope of ICCoh on log2(k) within each (dataset, method), so
       the velocities and the composition are held fixed in expectation and only
       the count moves. A slope near 0 means count per se does not bias ICCoh,
       whatever A shows. Count-matched ("min") draws also give a ranking at equal
       gene count.

    Slopes are per doubling of the gene count. CIs resample whole datasets.
    """
    empty = {}
    how = str(plot_cfg.get("gene_bias_aggregator")
              or plot_cfg.get("iccoh_aggregator", "median")).lower()
    how = "median" if how.startswith("med") else "mean"
    xkind = str(plot_cfg.get("gene_bias_x", "n_genes")).lower()
    B = _cfg_int(plot_cfg, "gene_bias_bootstrap", 2000)
    min_sd = _cfg_num(plot_cfg, "gene_bias_min_within_sd", 0.05)
    rng = np.random.default_rng(_cfg_int(plot_cfg, "gene_bias_seed", 0))

    if genes is None or genes.empty:
        genes = _genes_from_long(long_df)
    if gene_clusters is None or gene_clusters.empty:
        gene_clusters = _gene_clusters_from_long(long_df)
    if genes.empty or gene_clusters.empty:
        print("  Skipping the ICCoh gene-count analysis: no gene-space ICCoh gene "
              "tables (needs compute_cbdir_run.py with iccoh_space: gene_confidence)")
        return empty
    genes = genes[genes["method"].isin(method_order)].copy()
    gcl = gene_clusters[gene_clusters["method"].isin(method_order)].copy()

    lev = _collapse_clusters(gcl, ["dataset", "method", "gene_set", "kind", "k", "rep"], how)
    full = (lev[lev["kind"] == "full"]
            .pivot_table(index=["dataset", "method"], columns="gene_set",
                         values="iccoh", aggfunc="first").reset_index())
    full.columns.name = None
    full = full.rename(columns={"own": "iccoh_own", "shared": "iccoh_shared"})
    keep = [c for c in ("dataset", "method", "n_genes_own", "n_genes_shared", "pr_own",
                        "eff_genes_own", "pr_shared", "eff_genes_shared") if c in genes.columns]
    pts = genes[keep].drop_duplicates(["dataset", "method"]).merge(
        full, on=["dataset", "method"], how="inner")
    if "iccoh_own" not in pts.columns or pts["iccoh_own"].isna().all():
        print("  Skipping the ICCoh gene-count analysis: no own-gene ICCoh in the tables")
        return empty
    if xkind.startswith("eff") and "eff_genes_own" in pts.columns:
        pts["x"] = np.log2(pts["eff_genes_own"].astype(float))
        x_label = "log2 effective genes (own set)"
    else:
        pts["x"] = np.log2(pts["n_genes_own"].astype(float))
        x_label = "log2 genes scored (own set)"
        xkind = "n_genes"
    pts["aggregator"] = how
    tests, pm_rows, ranks = [], {}, []

    # ---- A. across methods ----------------------------------------------------
    a = pts.dropna(subset=["x", "iccoh_own"])
    for tag, fe_cols, note in (
            ("A1_between_methods", ["dataset"],
             "dataset fixed effects; confounded with method quality"),
            ("A2_within_method", ["dataset", "method"],
             "dataset + method fixed effects; within-method variation in gene count")):
        res = _fe_slope(a["iccoh_own"], a["x"], [a[c].to_numpy() for c in fe_cols],
                        min_sd) if len(a) >= 4 else None
        if res is None:
            tests.append(dict(test=tag, estimate=np.nan, se=np.nan, p=np.nan,
                              ci_low=np.nan, ci_high=np.nan, n_points=int(len(a)),
                              n_datasets=int(a["dataset"].nunique()),
                              note=f"not identified: < {min_sd:g} doublings of gene-count "
                                   f"spread left after the fixed effects"))
            continue
        lo, hi = _cluster_boot(
            a, lambda f, fc=fe_cols: (lambda r: None if r is None else r[0])(
                _fe_slope(f["iccoh_own"], f["x"], [f[c].to_numpy() for c in fc], min_sd)),
            B, rng)
        tests.append(dict(test=tag, estimate=res[0], se=res[1], p=res[2], ci_low=lo,
                          ci_high=hi, n_points=res[3], n_datasets=int(a["dataset"].nunique()),
                          note=note))

    # ---- B. own vs shared, paired ------------------------------------------------
    pb = pts.dropna(subset=["iccoh_own", "iccoh_shared", "n_genes_shared"]).copy() \
        if "iccoh_shared" in pts.columns else pd.DataFrame()
    if not pb.empty:
        pb["log2_ratio"] = np.log2(pb["n_genes_own"].astype(float)
                                   / pb["n_genes_shared"].astype(float))
        pb["delta"] = pb["iccoh_own"] - pb["iccoh_shared"]
        pts = pts.merge(pb[["dataset", "method", "log2_ratio", "delta"]],
                        on=["dataset", "method"], how="left")
        inf = pb[pb["log2_ratio"] > 1e-12]

        def _origin(f):
            xx = f["log2_ratio"].to_numpy(float)
            return float(np.sum(xx * f["delta"].to_numpy(float)) / np.sum(xx ** 2)) \
                if len(f) and np.sum(xx ** 2) > 0 else None
        if len(inf) >= 2:
            import statsmodels.api as sm
            fit = sm.OLS(inf["delta"].to_numpy(float), inf[["log2_ratio"]].to_numpy(float)).fit()
            lo, hi = _cluster_boot(inf, _origin, B, rng)
            tests.append(dict(test="B_own_vs_shared", estimate=float(fit.params[0]),
                              se=float(fit.bse[0]), p=float(fit.pvalues[0]), ci_low=lo,
                              ci_high=hi, n_points=int(len(inf)),
                              n_datasets=int(inf["dataset"].nunique()),
                              note=f"through the origin; {len(pb) - len(inf)} pair(s) with "
                                   f"own == shared shown but not fitted"))
        else:
            tests.append(dict(test="B_own_vs_shared", estimate=np.nan, se=np.nan, p=np.nan,
                              ci_low=np.nan, ci_high=np.nan, n_points=int(len(inf)),
                              n_datasets=int(inf["dataset"].nunique()) if len(inf) else 0,
                              note="fewer than two pairs where the own set is larger"))
        for m, g in pb.groupby("method"):
            gi = g[g["log2_ratio"] > 1e-12]
            lo, hi = _cluster_boot(gi, lambda f: float(f["delta"].mean()), B, rng) \
                if len(gi) else (np.nan, np.nan)
            pm_rows.setdefault(m, {}).update(
                mean_delta_own_minus_shared=float(gi["delta"].mean()) if len(gi) else 0.0,
                delta_ci_low=lo, delta_ci_high=hi,
                n_datasets_own_larger=int(gi["dataset"].nunique()))
        for d, g in pb.groupby("dataset"):
            if len(g) >= 3:
                tau = stats.kendalltau(g["iccoh_own"], g["iccoh_shared"])[0]
                ranks.append(dict(dataset=d, comparison="own_vs_shared",
                                  kendall_tau=float(tau), n_methods=int(len(g))))

    # ---- C. dose-response on random own-gene subsets ---------------------------
    dose = lev[lev["gene_set"].isin(["own", "own_subsample"])].copy()
    dose = (dose.groupby(["dataset", "method", "kind", "k"])["iccoh"].mean()
                .reset_index())
    dose = dose.merge(pts[["dataset", "method", "n_genes_own", "iccoh_own"]],
                      on=["dataset", "method"], how="left")
    dose["log2_k"] = np.log2(dose["k"].astype(float))
    dose["log2_frac"] = np.log2(dose["k"].astype(float) / dose["n_genes_own"].astype(float))
    dose["iccoh_centered"] = dose["iccoh"] - dose["iccoh_own"]
    slopes = []
    for (d, m), g in dose.groupby(["dataset", "method"]):
        g = g.dropna(subset=["iccoh", "log2_k"])
        if g["k"].nunique() >= 2:
            b = np.polyfit(g["log2_k"], g["iccoh"], 1)[0]
            slopes.append(dict(dataset=d, method=m, dose_slope=float(b),
                               n_sizes=int(g["k"].nunique())))
    slopes = pd.DataFrame(slopes)
    if not slopes.empty:
        per_ds = slopes.groupby("dataset")["dose_slope"].mean()
        tt = stats.ttest_1samp(per_ds.to_numpy(float), 0.0) if len(per_ds) >= 2 else None
        lo, hi = _cluster_boot(slopes, lambda f: float(f["dose_slope"].mean()), B, rng)
        tests.append(dict(test="C_dose_response", estimate=float(slopes["dose_slope"].mean()),
                          se=(float(per_ds.std(ddof=1) / np.sqrt(len(per_ds)))
                              if len(per_ds) >= 2 else np.nan),
                          p=(float(tt.pvalue) if tt is not None else np.nan),
                          ci_low=lo, ci_high=hi, n_points=int(len(slopes)),
                          n_datasets=int(slopes["dataset"].nunique()),
                          note="random subsets of each method's own genes; "
                               "t-test on per-dataset mean slopes"))
        for m, g in slopes.groupby("method"):
            lo, hi = _cluster_boot(g, lambda f: float(f["dose_slope"].mean()), B, rng)
            pm_rows.setdefault(m, {}).update(mean_dose_slope=float(g["dose_slope"].mean()),
                                             dose_ci_low=lo, dose_ci_high=hi)
        # Ranking at EQUAL gene count: each method's "min" draws (subsampled to
        # the dataset's smallest own-gene set); a method whose own set already
        # is the smallest has no draw and keeps its full-set score.
        mn = (dose[dose["kind"] == "min"].set_index(["dataset", "method"])["iccoh"]
              if (dose["kind"] == "min").any() else pd.Series(dtype=float))
        for d, g in pts.groupby("dataset"):
            if len(g) < 3 or not any((d, m) in mn.index for m in g["method"]):
                continue
            own = g.set_index("method")["iccoh_own"]
            matched = pd.Series({m: (mn[(d, m)] if (d, m) in mn.index else own[m])
                                 for m in own.index})
            tau = stats.kendalltau(own.to_numpy(float), matched[own.index].to_numpy(float))[0]
            ranks.append(dict(dataset=d, comparison="own_vs_count_matched",
                              kendall_tau=float(tau), n_methods=int(len(own))))
    else:
        tests.append(dict(test="C_dose_response", estimate=np.nan, se=np.nan, p=np.nan,
                          ci_low=np.nan, ci_high=np.nan, n_points=0, n_datasets=0,
                          note="no subsample draws (iccoh_gene_subsample was off)"))

    for m, g in pts.groupby("method"):
        pm_rows.setdefault(m, {}).update(
            n_datasets=int(g["dataset"].nunique()),
            mean_n_genes_own=float(g["n_genes_own"].mean()),
            mean_n_genes_shared=(float(g["n_genes_shared"].mean())
                                 if "n_genes_shared" in g else np.nan),
            mean_iccoh_own=float(g["iccoh_own"].mean()),
            mean_iccoh_shared=(float(g["iccoh_shared"].mean())
                               if "iccoh_shared" in g else np.nan))
    per_method = pd.DataFrame([dict(method=m, **v) for m, v in pm_rows.items()])
    if not per_method.empty:
        per_method["_o"] = per_method["method"].map({m: i for i, m in enumerate(method_order)})
        per_method = per_method.sort_values("_o").drop(columns="_o")
    tests = pd.DataFrame(tests)
    tests["x"] = xkind
    tests["aggregator"] = how

    print("\n=== Gene-count bias of gene-space ICCoh ===")
    print(f"    {len(pts)} (dataset, method) pairs; slopes are ICCoh per doubling of "
          f"the gene count ({x_label}); CIs resample datasets")
    for _, r in tests.iterrows():
        est = "   n/a" if not np.isfinite(r["estimate"]) else f"{r['estimate']:+.4f}"
        ci = ("" if not np.isfinite(r["ci_low"]) else
              f" [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
        pv = "" if not np.isfinite(r["p"]) else f"  p = {r['p']:.3g}"
        print(f"  {r['test']:<22s} {est}{ci}{pv}   ({r['note']})")
    if ranks:
        rk = pd.DataFrame(ranks)
        for comp, g in rk.groupby("comparison"):
            print(f"  ranking {comp:<22s} mean Kendall tau = {g['kendall_tau'].mean():+.3f} "
                  f"over {len(g)} dataset(s)")
    return dict(points=pts, tests=tests, per_method=per_method,
                ranks=pd.DataFrame(ranks), dose=dose, dose_slopes=slopes,
                x_label=x_label)


def plot_iccoh_gene_bias(res, method_order, plot_cfg, suffix):
    """Four panels, one question: is gene-space ICCoh tilted by gene count?

    a  across methods: ICCoh (own genes) vs log2 gene count, with the
       within-method slope (A2) drawn at the mean level.
    b  own minus shared genes vs log2(own / shared): what dropping to the shared
       set does to each method (B).
    c  dose-response: ICCoh on random own-gene subsets, centred on the full-set
       value, vs log2 of the fraction kept (C). Flat lines = no count bias.
    d  per-method dose-response slope with a dataset-bootstrap CI.
    """
    if not res or res.get("points") is None or res["points"].empty:
        return
    pts, tests = res["points"], res["tests"]
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    order = [m for m in method_order if m in set(pts["method"])]
    ds_names = list(pd.unique(pts["dataset"]))
    marker_map = _dataset_marker_map(ds_names, plot_cfg)
    ms = _cfg_num(plot_cfg, "gene_bias_marker_size", 7.0)
    t = tests.set_index("test")

    def _txt(tag, what):
        if tag not in t.index or not np.isfinite(t.loc[tag, "estimate"]):
            return f"{what}: not identified"
        r = t.loc[tag]
        ci = ("" if not np.isfinite(r["ci_low"]) else
              f" [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
        return f"{what}: {r['estimate']:+.3f}{ci}, p = {r['p']:.2g}"

    sns.set_theme(style="whitegrid")
    figsize = tuple(plot_cfg.get("gene_bias_figsize") or [13.0, 10.5])
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    def _scatter(ax, xcol, ycol, sub):
        for m in order:
            g = sub[sub["method"] == m]
            for d_name, gg in g.groupby("dataset"):
                ax.plot(gg[xcol], gg[ycol], marker=marker_map.get(d_name, "o"), ms=ms,
                        linestyle="none", color=palette.get(m, "#888888"), alpha=0.85,
                        markeredgecolor="#333333", markeredgewidth=0.4, zorder=3)

    # a
    ax = axes[0, 0]
    a = pts.dropna(subset=["x", "iccoh_own"])
    _scatter(ax, "x", "iccoh_own", a)
    if "A2_within_method" in t.index and np.isfinite(t.loc["A2_within_method", "estimate"]):
        b = float(t.loc["A2_within_method", "estimate"])
        xs = np.linspace(a["x"].min(), a["x"].max(), 20)
        ax.plot(xs, a["iccoh_own"].mean() + b * (xs - a["x"].mean()), ls="--", lw=1.6,
                color="#111111", zorder=4)
    ax.set_xlabel(res.get("x_label", "log2 genes scored"), fontsize=10.5, fontweight="bold")
    ax.set_ylabel("ICCoh on own genes", fontsize=10.5, fontweight="bold")
    ax.set_title("a  Across methods and datasets", fontsize=11, fontweight="bold", loc="left")
    ax.text(0.015, 0.985, _txt("A2_within_method", "within-method slope") + "\n"
            + _txt("A1_between_methods", "between-method slope"),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.3, color="#111111")

    # b
    ax = axes[0, 1]
    if "delta" in pts.columns and pts["delta"].notna().any():
        pb = pts.dropna(subset=["delta"])
        _scatter(ax, "log2_ratio", "delta", pb)
        if "B_own_vs_shared" in t.index and np.isfinite(t.loc["B_own_vs_shared", "estimate"]):
            b = float(t.loc["B_own_vs_shared", "estimate"])
            xs = np.linspace(0, max(pb["log2_ratio"].max(), 1e-3), 20)
            ax.plot(xs, b * xs, ls="--", lw=1.6, color="#111111", zorder=4)
        draw_zero_line(ax, plot_cfg)
        ax.text(0.015, 0.985, _txt("B_own_vs_shared", "slope through origin"),
                transform=ax.transAxes, va="top", ha="left", fontsize=8.3, color="#111111")
    else:
        ax.text(0.5, 0.5, "no shared-gene ICCoh\n(iccoh_gene_set_contrast off)",
                ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.set_xlabel("log2(own genes / shared genes)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("ICCoh own − ICCoh shared", fontsize=10.5, fontweight="bold")
    ax.set_title("b  Own vs shared genes (same velocities)", fontsize=11,
                 fontweight="bold", loc="left")

    # c
    ax = axes[1, 0]
    dose = res.get("dose", pd.DataFrame())
    if dose is not None and not dose.empty and dose["kind"].nunique() > 1:
        for (d_name, m), g in dose.groupby(["dataset", "method"]):
            if m not in order:
                continue
            g = g.sort_values("log2_frac")
            ax.plot(g["log2_frac"], g["iccoh_centered"], "-", lw=1.0, alpha=0.55,
                    color=palette.get(m, "#888888"), zorder=2)
            ax.plot(g["log2_frac"], g["iccoh_centered"], linestyle="none",
                    marker=marker_map.get(d_name, "o"), ms=ms * 0.7,
                    color=palette.get(m, "#888888"), markeredgecolor="#333333",
                    markeredgewidth=0.3, alpha=0.85, zorder=3)
        draw_zero_line(ax, plot_cfg)
        ax.text(0.015, 0.985, _txt("C_dose_response", "mean slope"),
                transform=ax.transAxes, va="top", ha="left", fontsize=8.3, color="#111111")
    else:
        ax.text(0.5, 0.5, "no subsample draws\n(iccoh_gene_subsample off)",
                ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.set_xlabel("log2(fraction of own genes kept)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("ICCoh − ICCoh on all own genes", fontsize=10.5, fontweight="bold")
    ax.set_title("c  Dose-response: random own-gene subsets", fontsize=11,
                 fontweight="bold", loc="left")

    # d
    ax = axes[1, 1]
    pm = res.get("per_method", pd.DataFrame())
    if pm is not None and "mean_dose_slope" in pm.columns:
        rows = [m for m in order if m in set(pm.dropna(subset=["mean_dose_slope"])["method"])]
        rows = rows[::-1]
        for yi, m in enumerate(rows):
            r = pm[pm["method"] == m].iloc[0]
            if np.isfinite(r.get("dose_ci_low", np.nan)):
                ax.hlines(yi, r["dose_ci_low"], r["dose_ci_high"], color="#52514e", lw=2.0)
            ax.plot([r["mean_dose_slope"]], [yi], "o", ms=9, color=palette.get(m, "#888888"),
                    markeredgecolor="#111111", markeredgewidth=0.8, zorder=3)
        draw_zero_line(ax, plot_cfg, vertical=False)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
    else:
        ax.text(0.5, 0.5, "no subsample draws", ha="center", va="center",
                transform=ax.transAxes, color="#666666")
    ax.set_xlabel("ICCoh change per doubling of genes", fontsize=10.5, fontweight="bold")
    ax.set_title("d  Count effect per method (95% CI, dataset bootstrap)", fontsize=11,
                 fontweight="bold", loc="left")

    from matplotlib.lines import Line2D
    m_handles = [Line2D([], [], marker="o", color=palette.get(m, "#888888"), linestyle="none",
                        markersize=6, markeredgecolor="#333333", markeredgewidth=0.4, label=m)
                 for m in order]
    d_handles = [Line2D([], [], marker=marker_map[dn], color="#555555", linestyle="none",
                        markersize=6, markeredgecolor="#333333", markeredgewidth=0.4,
                        label=str(dn)) for dn in ds_names]
    ncol_m = min(len(m_handles), 5) or 1
    ncol_d = min(len(d_handles), 4) or 1
    fig.legend(handles=m_handles, title="Method", fontsize=8, title_fontsize=9,
               loc="lower center", bbox_to_anchor=(0.28, 0.005), frameon=False, ncol=ncol_m)
    fig.legend(handles=d_handles, title="Dataset", fontsize=8, title_fontsize=9,
               loc="lower center", bbox_to_anchor=(0.78, 0.005), frameon=False, ncol=ncol_d)
    leg_rows = max(int(np.ceil(len(m_handles) / ncol_m)), int(np.ceil(len(d_handles) / ncol_d)))
    fig.suptitle(plot_cfg.get("gene_bias_title",
                              "Does the number of genes scored bias gene-space ICCoh?"),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03 + 0.03 * (leg_rows + 0.5), 1, 0.95])
    _save_figure(f"{save_base}_iccoh_gene_bias{suffix}", dpi)


def fit_stability_per_dataset(levels, method_order, plot_cfg):
    """Level and consistency WITHIN each dataset, transition by transition.

    The cross-dataset table asks "does this method work everywhere". This one
    asks the question underneath it: inside one dataset, is the method ahead on
    every transition or carried by one easy edge, and which datasets is it ahead
    on at all. The replication unit is therefore the TRANSITION within a dataset,
    and the paired Wilcoxon against the reference is computed over that dataset's
    transitions alone.

    CAUTION, and it is the whole reason this table reports counts beside the p:
    datasets here carry a handful of transitions, so the exact signed-rank test
    has almost no resolution — with n transitions the smallest attainable
    two-sided p is 2 / 2**n (0.25 at n = 3, 0.125 at n = 4, 0.0625 at n = 5).
    A dataset with fewer than `stability_per_dataset_min_transitions` paired
    transitions is not tested at all rather than tested badly, and even where a
    p appears it should be read as a direction-and-count summary, not as
    inference. `n_transitions_positive` and the sign of `median_diff_vs_ref` are
    the honest per-dataset statements; the pooled tests elsewhere carry the
    inference.
    """
    if levels is None or levels.empty:
        return pd.DataFrame()
    methods = [m for m in method_order if m in levels.columns] or list(levels.columns)
    ref = plot_cfg.get("reference_method")
    alt = str(plot_cfg.get("stability_wilcoxon_alternative", "two-sided")).lower()
    zero_method = str(plot_cfg.get("stability_wilcoxon_zero_method", "wilcox"))
    min_e = _cfg_int(plot_cfg, "stability_per_dataset_min_transitions", 3)
    p_adjust = str(plot_cfg.get("stability_p_adjust")
                   or plot_cfg.get("p_adjust", "fdr_bh")).lower()
    scope = str(plot_cfg.get("stability_per_dataset_p_adjust_scope", "dataset")).lower()
    transform = str(plot_cfg.get("transform", "atanh")).lower()

    rows = []
    for d_name, blk in levels.groupby(level="dataset", sort=False):
        blk = blk.droplevel("dataset")
        for m in methods:
            col = blk[m].dropna() if m in blk.columns else pd.Series(dtype=float)
            if col.empty:
                continue
            n_pos = int((col > 0).sum())
            rec = dict(
                dataset=str(d_name), method=m, reference_method=(ref or ""),
                n_transitions=int(col.size), n_transitions_positive=n_pos,
                mean_z=float(col.mean()), median_z=float(col.median()),
                sd_transitions=float(col.std(ddof=1)) if col.size > 1 else np.nan,
                worst_transition=float(col.min()), best_transition=float(col.max()),
                worst_transition_name=str(col.idxmin()),
                best_transition_name=str(col.idxmax()),
                n_transitions_paired=np.nan, median_diff_vs_ref=np.nan,
                w_stat_vs_ref=np.nan, p_wilcoxon_vs_ref=np.nan)
            rec["p_transitions_positive"] = float(
                stats.binomtest(n_pos, int(col.size), 0.5, "greater").pvalue)
            for src, dst in (("mean_z", "mean_native"),
                             ("median_z", "median_native"),
                             ("worst_transition", "worst_transition_native"),
                             ("best_transition", "best_transition_native")):
                rec[dst] = float(_inverse_transform(np.array([rec[src]]), transform)[0])
            if ref and ref in blk.columns and m != ref:
                pair = blk[[m, ref]].dropna()
                diff = (pair[m] - pair[ref]).to_numpy(dtype=float)
                rec["n_transitions_paired"] = int(diff.size)
                rec["median_diff_vs_ref"] = (float(np.median(diff))
                                             if diff.size else np.nan)
                rec["n_transitions_above_ref"] = int((diff > 0).sum())
                if diff.size >= min_e and np.any(diff != 0):
                    try:
                        w, p = stats.wilcoxon(pair[m].to_numpy(dtype=float),
                                              pair[ref].to_numpy(dtype=float),
                                              alternative=alt, zero_method=zero_method)
                        rec["w_stat_vs_ref"] = float(w)
                        rec["p_wilcoxon_vs_ref"] = float(p)
                    except Exception as e:
                        print(f"  WARNING: per-dataset Wilcoxon failed for "
                              f"{m} on {d_name}: {type(e).__name__}: {e}")
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # FDR family: by default the methods within one dataset, because that is the
    # family one panel displays. "global" adjusts over every (dataset, method)
    # cell at once, which is stricter and right if the whole grid is read as one
    # screen of comparisons.
    out["p_wilcoxon_vs_ref_adj"] = np.nan
    groups = ([(None, out.index)] if scope.startswith("g")
              else list(out.groupby("dataset").groups.items()))
    for _, idx in groups:
        p = out.loc[idx, "p_wilcoxon_vs_ref"].to_numpy(dtype=float)
        ok = np.isfinite(p)
        if not ok.any():
            continue
        adj = np.full(p.shape, np.nan)
        if p_adjust == "none":
            adj[ok] = p[ok]
        else:
            from statsmodels.stats.multitest import multipletests
            adj[ok] = multipletests(p[ok], method=p_adjust)[1]
        out.loc[idx, "p_wilcoxon_vs_ref_adj"] = adj
    out["stability_p_adjust"] = p_adjust
    out["p_adjust_scope"] = ("global" if scope.startswith("g") else "dataset")

    print("\n=== Per-dataset level and consistency (unit = transition) ===")
    if ref:
        print(f"    paired Wilcoxon vs {ref} within each dataset; with a handful of "
              f"transitions the p has little resolution — read the counts")
    for d_name, blk in out.groupby("dataset", sort=False):
        print(f"  {d_name}:")
        for _, r in blk.sort_values("mean_z", ascending=False).iterrows():
            tail = ""
            if r["method"] == ref:
                tail = "  reference"
            elif np.isfinite(r["p_wilcoxon_vs_ref"]):
                tail = (f"  vs ref: median diff={r['median_diff_vs_ref']:+.4f}  "
                        f"p_adj={r['p_wilcoxon_vs_ref_adj']:.3g}")
            elif np.isfinite(r["median_diff_vs_ref"]):
                tail = (f"  vs ref: median diff={r['median_diff_vs_ref']:+.4f}  "
                        f"(too few transitions to test)")
            print(f"    {r['method']:<20s} mean={r['mean_z']:+.4f}  "
                  f"worst={r['worst_transition']:+.4f} ({r['worst_transition_name']})  "
                  f"{int(r['n_transitions_positive'])}/{int(r['n_transitions'])} "
                  f"{UNIT_LABEL_PLURAL} above zero{tail}")
    return out


def _format_p_compact(p):
    """Short p-value text for in-panel annotation."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    if p < 0.001:
        return f"{p:.1e}".replace("e-0", "e-")
    return f"{p:.3f}".rstrip("0").rstrip(".") if p < 1 else "1"


def _dataset_marker_map(datasets, plot_cfg):
    """Stable dataset -> marker shape assignment for the stability dot panel."""
    default = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*"]
    cycle = plot_cfg.get("stability_dataset_marker_list") or default
    cycle = [str(s) for s in cycle] or default
    explicit = plot_cfg.get("stability_dataset_markers_map") or {}
    n_free = len([d for d in datasets if d not in explicit])
    if n_free > len(cycle):
        print(f"  WARNING: {n_free} units but only {len(cycle)} marker shapes — "
              f"shapes repeat; extend stability_dataset_marker_list or turn the "
              f"shapes off for this figure")
    out, i = {}, 0
    for d in datasets:
        if d in explicit:
            out[d] = str(explicit[d])
        else:
            out[d] = cycle[i % len(cycle)]
            i += 1
    return out


def plot_stability(stability, levels, method_order, plot_cfg, suffix,
                   dataset_order=None):
    """Level and consistency in one frame — the main-text consistency panel.

    Left: one dot per dataset for every method, the mean marked, and the worst
    dataset drawn larger and labelled. A method whose dots all sit right of zero
    has never failed a dataset; that is the claim, shown rather than asserted.
    With `stability_dataset_markers`, each dataset gets its own marker shape, so
    a reader can tell which dataset a low dot came from and whether the same
    dataset is the weak one for every method. The row label carries the
    dataset-level paired Wilcoxon against the reference (FDR-adjusted) when that
    test ran, so the panel's visual claim and its test sit on the same line.
    Right (optional): mean against between-dataset SD, with mean - SD iso-lines,
    so "high and stable" is a direction on the plot rather than two numbers.
    """
    if stability is None or stability.empty or levels is None or levels.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    spec = plot_cfg.get("stability_panels", ["dataset_dots", "mean_sd"])
    S = stability.set_index("method")
    how = str(plot_cfg.get("stability_order", "worst_dataset")).lower()
    if how == "specified":
        order = [m for m in method_order if m in S.index]
    else:
        key = "mean_z" if how == "mean" else "worst_dataset"
        order = list(S[key].sort_values(ascending=False).index)

    by_shape = bool(plot_cfg.get("stability_dataset_markers", False))
    ds_means = levels.groupby(level="dataset").mean()

    def _label(r):
        return (f"{int(r['n_datasets_positive'])}/{int(r['n_datasets'])}"
                f" · {int(r['n_transitions_positive'])}/{int(r['n_transitions'])}")

    _draw_stability_figure(
        ds_means, S, order, plot_cfg, spec,
        palette=build_palette(method_order, plot_cfg.get("method_colors")),
        out_base=f"{save_base}_stability{suffix}",
        unit_singular="dataset", unit_plural="datasets",
        x_label_unit="dataset",
        dot_title="Every dataset, every method",
        label_fn=_label,
        label_note="labels = datasets · transitions above zero",
        highlight_fn=lambda r: r["n_datasets_positive"] == r["n_datasets"],
        sd_col="sd_datasets", mean_col="mean_of_dataset_means",
        sd_label="SD across datasets  (lower = more consistent)",
        mean_label=f"Mean {VALUE_LABEL} across datasets (Fisher z)",
        by_shape=by_shape, legend_title="Dataset",
        legend_loc=plot_cfg.get("stability_dataset_legend_loc", "outside"),
        suptitle=plot_cfg.get("stability_title",
                              "Consistency across datasets, not just average level"))

    if plot_cfg.get("stability_summary_plot", True):
        _draw_stability_summary(
            ds_means, order, plot_cfg,
            palette=build_palette(method_order, plot_cfg.get("method_colors")),
            heat_base=f"{save_base}_stability_heatmap{suffix}",
            scatter_base=f"{save_base}_stability_mean_vs_sd{suffix}",
            unit_singular="dataset", unit_plural="datasets",
            col_order=dataset_order,
            heat_title=plot_cfg.get("stability_summary_heatmap_title"),
            scatter_title=plot_cfg.get("stability_summary_scatter_title"))


def plot_stability_per_dataset(per_dataset, levels, method_order, plot_cfg, suffix,
                               dataset_order=None):
    """One stability frame per dataset, with a dot per transition.

    Same frame as the cross-dataset figure, one level down: within a dataset,
    each point is a transition, so a method that wins the dataset on one easy
    edge looks different from one that is ahead on all of them. Read beside the
    cross-dataset figure it answers "where is the method better or worse", which
    a single pooled mean cannot.
    """
    if per_dataset is None or per_dataset.empty or levels is None or levels.empty:
        return []
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    sub = plot_cfg.get("stability_per_dataset_subdir", "stability_per_dataset")
    out_dir = os.path.dirname(os.path.abspath(save_base))
    stem = os.path.basename(save_base)
    if sub:
        out_dir = os.path.join(out_dir, str(sub))
    os.makedirs(out_dir, exist_ok=True)

    spec = (plot_cfg.get("stability_per_dataset_panels")
            or plot_cfg.get("stability_panels", ["dataset_dots", "mean_sd"]))
    by_shape = bool(plot_cfg.get("stability_per_dataset_markers",
                                 plot_cfg.get("stability_dataset_markers", False)))
    how = str(plot_cfg.get("stability_per_dataset_order",
                           plot_cfg.get("stability_order", "worst_dataset"))).lower()
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    names = list(dataset_order) if dataset_order else []
    names += [d for d in per_dataset["dataset"].unique() if d not in names]

    written = []
    for d_name in names:
        blk = per_dataset[per_dataset["dataset"] == d_name]
        if blk.empty or d_name not in levels.index.get_level_values("dataset"):
            continue
        vals = levels.xs(d_name, level="dataset")
        S = blk.set_index("method")
        S = S.loc[[m for m in method_order if m in S.index]
                  or list(S.index)]
        if how == "specified":
            order = list(S.index)
        else:
            key = "mean_z" if how == "mean" else "worst_transition"
            order = list(S[key].sort_values(ascending=False).index)
        n_e = int(S["n_transitions"].max()) if len(S) else 0

        def _label(r):
            return (f"{int(r['n_transitions_positive'])}/{int(r['n_transitions'])}")

        _draw_stability_figure(
            vals, S, order, plot_cfg, spec, palette=palette,
            out_base=os.path.join(out_dir, f"{stem}_stability_{_slug(d_name)}{suffix}"),
            unit_singular=UNIT_LABEL, unit_plural=UNIT_LABEL_PLURAL,
            x_label_unit=UNIT_LABEL,
            dot_title=f"{d_name}: every {UNIT_LABEL}, every method",
            label_fn=_label,
            label_note=f"labels = {UNIT_LABEL_PLURAL} above zero",
            highlight_fn=lambda r: r["n_transitions_positive"] == r["n_transitions"],
            sd_col="sd_transitions", mean_col="mean_z",
            sd_label=f"SD across {UNIT_LABEL_PLURAL}  (lower = more consistent)",
            mean_label=f"Mean {VALUE_LABEL} across {UNIT_LABEL_PLURAL} (Fisher z)",
            by_shape=by_shape, legend_title=UNIT_LABEL.capitalize(),
            legend_loc=plot_cfg.get("stability_per_dataset_legend_loc", "outside"),
            suptitle=plot_cfg.get("stability_per_dataset_title")
            or f"{d_name} — consistency across its {n_e} "
               f"{UNIT_LABEL_PLURAL if n_e != 1 else UNIT_LABEL}")
        if plot_cfg.get("stability_per_dataset_summary_plot",
                        plot_cfg.get("stability_summary_plot", True)):
            slug = _slug(d_name)
            _draw_stability_summary(
                vals, order, plot_cfg, palette=palette,
                heat_base=os.path.join(out_dir, f"{stem}_stability_heatmap_{slug}{suffix}"),
                scatter_base=os.path.join(out_dir,
                                          f"{stem}_stability_mean_vs_sd_{slug}{suffix}"),
                unit_singular=UNIT_LABEL, unit_plural=UNIT_LABEL_PLURAL,
                heat_title=d_name, scatter_title=d_name, prettify_cols=False)
        written.append(d_name)
    return written


def _draw_stability_figure(vals, S, order, plot_cfg, spec, *, palette, out_base,
                           unit_singular, unit_plural, x_label_unit, dot_title,
                           label_fn, label_note, highlight_fn, sd_col, mean_col,
                           sd_label, mean_label, by_shape, legend_title, legend_loc,
                           suptitle):
    """Draw the stability frame for one replication unit (dataset or transition).

    `vals` is a unit x method matrix on the Fisher-z scale — dataset means for
    the cross-dataset figure, a single dataset's transition means for the
    per-dataset ones. Everything that differs between the two is passed in, so
    the two figures cannot drift apart as either is tweaked.
    """
    dpi = plot_cfg.get("dpi", 300)
    transform = str(plot_cfg.get("transform", "atanh")).lower()
    native = str(plot_cfg.get("stability_scale", "native")).lower().startswith("n")
    if isinstance(spec, str):
        spec = [spec]
    panels = [str(s).strip().lower() for s in spec
              if str(s).strip().lower() in ("dataset_dots", "mean_sd")]
    if not panels:
        return

    rows = order[::-1]                      # best at the top of a horizontal axis
    conv = (lambda v: _inverse_transform(np.asarray(v, dtype=float), transform)
            ) if native else (lambda v: np.asarray(v, dtype=float))
    unit = VALUE_LABEL if native else f"{VALUE_LABEL} (Fisher z)"

    unit_names = list(vals.index)
    marker_map = _dataset_marker_map(unit_names, plot_cfg) if by_shape else {}
    ref_method = plot_cfg.get("reference_method")
    show_p = (bool(plot_cfg.get("stability_annotate_p_adj", True))
              and "p_wilcoxon_vs_ref_adj" in S.columns
              and S["p_wilcoxon_vs_ref_adj"].notna().any())

    figsize = tuple(plot_cfg.get("stability_figsize")
                    or [6.4 * len(panels), 0.46 * max(len(rows), 3) + 2.6])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, dpi=dpi, squeeze=False)
    dot_ax, notes = None, []

    for ax, panel in zip(axes[0], panels):
        if panel == "dataset_dots":
            dot_ax = ax
            for y, m in enumerate(rows):
                ser = vals[m].dropna() if m in vals.columns else pd.Series(dtype=float)
                v_arr = conv(ser.to_numpy()) if ser.size else np.array([])
                col = palette.get(m, "#888888")
                if v_arr.size:
                    ax.plot([v_arr.min(), v_arr.max()], [y, y], color=col, lw=1.2,
                            alpha=0.45, zorder=1, solid_capstyle="round")
                    worst = int(np.argmin(v_arr))
                    if by_shape:
                        # One shape per unit, so a dot can be traced back to the
                        # dataset (or transition) it came from without a second figure.
                        for i, (u_name, v) in enumerate(zip(ser.index, v_arr)):
                            big = (i == worst)
                            ax.plot([v], [y], marker=marker_map.get(u_name, "o"),
                                    color=col, ms=(9.5 if big else 6),
                                    alpha=(1.0 if big else 0.8), linestyle="none",
                                    markeredgecolor=("#111111" if big else "#333333"),
                                    markeredgewidth=(1.2 if big else 0.4),
                                    zorder=(3 if big else 2))
                    else:
                        ax.plot(v_arr, np.full(v_arr.size, y), "o", color=col, ms=5,
                                alpha=0.75, markeredgecolor="#333333",
                                markeredgewidth=0.4, zorder=2)
                        ax.plot([v_arr.min()], [y], "o", color=col, ms=10, zorder=3,
                                markeredgecolor="#111111", markeredgewidth=1.2)
                    ax.plot([float(np.mean(v_arr))], [y], "|", color="#111111",
                            ms=16, mew=2.0, zorder=4)
                if plot_cfg.get("stability_annotate", True) and m in S.index:
                    r = S.loc[m]
                    txt = "   " + label_fn(r)
                    if show_p:
                        txt += (" · ref" if m == ref_method else
                                " · " + _format_p_compact(
                                    r.get("p_wilcoxon_vs_ref_adj", np.nan)))
                    hot = bool(highlight_fn(r))
                    notes.append(ax.text(v_arr.max() if v_arr.size else 0.0, y, txt,
                                         va="center", fontsize=8,
                                         color=("#c2410c" if hot else "#777777"),
                                         fontweight=("bold" if hot else "normal")))
            draw_zero_line(ax, plot_cfg, vertical=False)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows)
            ax.set_xlabel(f"Mean {unit} per {x_label_unit}", fontsize=11,
                          fontweight="bold")
            ax.set_title(dot_title + "\n"
                         + (f"shape = {unit_singular},  outlined = worst "
                            f"{unit_singular},  tick = mean" if by_shape else
                            f"large dot = worst {unit_singular},  tick = mean")
                         + "\n" + label_note + (" · p-adj" if show_p else ""),
                         fontsize=11, fontweight="bold")
            if by_shape and legend_loc != "none":
                from matplotlib.lines import Line2D
                handles = [Line2D([], [], marker=marker_map[u], color="#555555",
                                  linestyle="none", markersize=6,
                                  markeredgecolor="#333333", markeredgewidth=0.4,
                                  label=str(u)) for u in unit_names]
                if str(legend_loc).lower() == "outside":
                    # No legend title below the axis: it lands on the x label.
                    ax.legend(handles=handles, fontsize=7, loc="upper left",
                              bbox_to_anchor=(0.0, -0.20),
                              ncol=min(len(handles), 4), frameon=False)
                else:
                    ax.legend(handles=handles, title=legend_title, fontsize=7,
                              title_fontsize=8, loc=legend_loc, frameon=True,
                              framealpha=0.9,
                              ncol=_cfg_int(plot_cfg,
                                            "stability_dataset_legend_ncol", 1))
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo, hi + (0.30 if notes else 0.02) * (hi - lo))
        else:
            x = S.loc[order, sd_col].to_numpy(dtype=float)
            y = S.loc[order, mean_col].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.any():
                span = max(x[ok].max(), 1e-6)
                grid = np.linspace(0, span * 1.15, 50)
                for c in np.round(np.linspace(np.nanmin(y - x), np.nanmax(y), 5), 3):
                    ax.plot(grid, c + grid, ls=":", lw=0.8, color="#bbbbbb", zorder=0)
                for m, xi, yi in zip(order, x, y):
                    if not (np.isfinite(xi) and np.isfinite(yi)):
                        continue
                    ax.plot([xi], [yi], "o", color=palette.get(m, "#888888"), ms=9,
                            markeredgecolor="#333333", markeredgewidth=0.6, zorder=3)
                    ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(7, 4),
                                fontsize=8, color="#333333")
            draw_zero_line(ax, plot_cfg)     # below it, the method is net-reversed
            ax.set_xlabel(sd_label, fontsize=11, fontweight="bold")
            ax.set_ylabel(mean_label, fontsize=11, fontweight="bold")
            ax.set_title("High and stable is up and to the left\n"
                         "dotted lines are constant mean − SD",
                         fontsize=11, fontweight="bold")
            ax.set_xlim(left=0)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    # The annotations are anchored in data coordinates, so widen the axis until
    # they fit rather than reserving a guessed gutter that is usually wrong.
    if dot_ax is not None and notes:
        for _ in range(4):
            try:
                fig.canvas.draw()
                rend = fig.canvas.get_renderer()
                box = dot_ax.get_window_extent(renderer=rend)
                right = max(t.get_window_extent(renderer=rend).x1 for t in notes)
                lo_c, hi_c = dot_ax.get_xlim()
                over = right - (box.x1 - 3)
                if abs(over) < 4:
                    break
                dot_ax.set_xlim(lo_c, hi_c + over * (hi_c - lo_c) / max(box.width, 1.0))
            except Exception:
                break
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_figure(out_base, dpi)


# ---------------------------------------------------------------------------
# Stability, summary version: a heatmap of every (method, dataset) mean beside
# its mean ± SD, and a clean mean-vs-SD scatter. Same numbers as the dot-panel
# figure, drawn for a reader who wants to look a value up rather than read a
# distribution.
# ---------------------------------------------------------------------------

# Nine-step diverging scale, blue (reversed) - grey (no direction) - red (correct).
_SUMMARY_DIVERGING = ["#5a9be0", "#83b3e8", "#abcbef", "#d4e4f6", "#ebebeb",
                      "#f6d6d8", "#eeb1b5", "#e68b91", "#de656e"]
_SUMMARY_SEQUENTIAL = ["#f3f6fa", "#dfe9f3", "#c6daeb", "#a9c7e1", "#89b2d4",
                       "#6a9cc6", "#4e85b6", "#386ea3", "#26588e"]
_MINUS = "−"


def _summary_value_phrase(plot_cfg):
    """Plain-language name of the plotted value for axis labels and legends.

    `stability_summary_value_label` overrides it. Otherwise CBDir reads as
    "correctness of transitions (CBDir)", and anything else (the alignment pass,
    velocity confidence reusing this module) keeps its own VALUE_LABEL.
    """
    custom = plot_cfg.get("stability_summary_value_label")
    if custom:
        return str(custom)
    known = {
        "cbdir": "correctness of transitions (CBDir)",
        "alignment": "correctness of transitions, scaled to its ceiling (Alignment)",
    }
    return known.get(str(VALUE_LABEL).strip().lower(), str(VALUE_LABEL))


def _fmt_signed(v, nd=2):
    """`0.12` / `−0.12` with a true minus sign, `–` for missing."""
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{v:.{nd}f}"
    if s.startswith("-"):
        s = _MINUS + s[1:]
        if float(v) > -0.5 * 10 ** (-nd):     # "-0.00" -> "0.00"
            s = s[1:]
    return s


def _summary_names(plot_cfg, key, names):
    """Display names from an optional {raw: shown} map in the config."""
    m = plot_cfg.get(key) or {}
    return {n: str(m.get(n, n)) for n in names}


def _summary_highlight(plot_cfg):
    h = plot_cfg.get("stability_summary_highlight")
    if h is None:
        return set()
    return {str(x) for x in (h if isinstance(h, (list, tuple)) else [h])}


def _place_labels(ax, fig, pts, texts, styles, pad_px=3.0):
    """Greedy, dependency-free label placement for a small scatter.

    Each label tries eight positions around its point, nearest first, and takes
    the first that overlaps neither another label nor another marker and stays
    inside the axes. With ~10 methods this settles every label without the
    leader lines a general-purpose solver would add.
    """
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer=rend)
    disp = ax.transData.transform(np.asarray(pts, dtype=float))
    r_mk = 7.0
    marker_boxes = [Bbox.from_extents(x - r_mk, y - r_mk, x + r_mk, y + r_mk)
                    for x, y in disp]
    cands = [(9, 0, "left", "center"), (-9, 0, "right", "center"),
             (0, 10, "center", "bottom"), (0, -10, "center", "top"),
             (8, 8, "left", "bottom"), (8, -8, "left", "top"),
             (-8, 8, "right", "bottom"), (-8, -8, "right", "top")]
    placed = []
    for i, ((x, y), txt, sty) in enumerate(zip(pts, texts, styles)):
        best = None
        for dx, dy, ha, va in cands:
            a = ax.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va=va, **sty)
            bb = a.get_window_extent(renderer=rend).expanded(1.0, 1.0)
            bb = Bbox.from_extents(bb.x0 - pad_px, bb.y0 - pad_px,
                                   bb.x1 + pad_px, bb.y1 + pad_px)
            inside = (bb.x0 >= ax_box.x0 and bb.x1 <= ax_box.x1
                      and bb.y0 >= ax_box.y0 and bb.y1 <= ax_box.y1)
            hits = sum(bb.overlaps(o) for o in placed) * 10
            hits += sum(bb.overlaps(mb) for j, mb in enumerate(marker_boxes) if j != i)
            score = hits + (0 if inside else 5)
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, a, bb)
            else:
                a.remove()
            if score == 0:
                break
        placed.append(best[2])


def _draw_stability_summary(vals, order, plot_cfg, *, palette, heat_base, scatter_base,
                            unit_singular, unit_plural, col_order=None,
                            heat_title=None, scatter_title=None, prettify_cols=True,
                            cluster_cols=None, raw=False, phrase=None,
                            scale="diverging", count_col=True, zero_ref=True,
                            direction_words=("worse", "better"), best_note=None,
                            vrange=None, cell_stat="Mean"):
    """Heatmap + mean ± SD, and a mean-vs-SD scatter, for one replication unit.

    `vals` is the same unit x method matrix (Fisher z) the dot-panel figure
    draws. Every number on both figures is computed from the values printed in
    the heatmap cells — tanh-back-transformed when `stability_scale` is native —
    so a reader can check the mean and SD in the margin against the row beside
    it. (The *_stability.csv summaries are on the z scale and differ slightly.)

    The keyword block after `cluster_cols` lets a metric that is not CBDir reuse
    the figure: `raw` skips the back-transform, `scale="sequential"` swaps the
    diverging scale for a one-hue ramp (for a metric where 0 is not a boundary,
    such as ICCoh), `count_col` / `zero_ref` drop the "> 0" column and the zero
    rules, and `direction_words` renames the worse/better arrows.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.gridspec import GridSpec
    import matplotlib.patheffects as pe

    dpi = plot_cfg.get("dpi", 300)
    transform = str(plot_cfg.get("transform", "atanh")).lower()
    native = str(plot_cfg.get("stability_scale", "native")).lower().startswith("n")
    V = vals.copy()
    if native and not raw:
        V = V.apply(lambda c: pd.Series(_inverse_transform(c.to_numpy(dtype=float),
                                                           transform), index=c.index))
    methods_all = [m for m in order if m in V.columns]
    if not methods_all or V.empty:
        return
    cols = [c for c in (col_order or []) if c in V.index]
    cols += [c for c in V.index if c not in cols]
    V = V.loc[cols, methods_all]

    # Optional hierarchical clustering of the heatmap columns (datasets, or
    # transitions in the per-dataset figures). Each column is described by its
    # vector of method values, so columns land together when the methods score
    # alike on them. Display order only: no number on the figure changes.
    if cluster_cols is None:
        cluster_cols = plot_cfg.get("stability_summary_cluster_columns", True)
    col_link = None
    if cluster_cols and V.shape[0] >= 3:
        try:
            from scipy.cluster import hierarchy as sch
            from scipy.spatial.distance import pdist
            X = V.to_numpy(dtype=float)
            X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0, keepdims=True))
            X = np.nan_to_num(X)
            metric = str(plot_cfg.get("stability_summary_cluster_metric",
                                      "euclidean")).lower()
            if metric == "correlation" and np.any(X.std(axis=1) == 0):
                metric = "euclidean"            # a flat column has no correlation
            D = np.clip(np.nan_to_num(pdist(X, metric=metric)), 0, None)
            col_link = sch.linkage(D, method=str(plot_cfg.get(
                "stability_summary_cluster_linkage", "average")), optimal_ordering=True)
            leaves = sch.leaves_list(col_link)
            cols = [cols[i] for i in leaves]
            V = V.loc[cols]
        except Exception as e:
            print(f"  WARNING: column clustering skipped: {type(e).__name__}: {e}")
            col_link = None

    mean = V.mean(axis=0)
    sd = V.std(axis=0, ddof=1)
    n_pos = (V > 0).sum(axis=0)
    n_obs = V.notna().sum(axis=0)

    how = str(plot_cfg.get("stability_summary_order", "mean")).lower()
    if how == "mean":
        rows = list(mean.sort_values(ascending=False).index)
    elif how == "mean_minus_sd":
        rows = list((mean - sd.fillna(0)).sort_values(ascending=False).index)
    else:                                   # "stability": same rows as the dot panel
        rows = methods_all

    hl = _summary_highlight(plot_cfg)
    hl_suffix = str(plot_cfg.get("stability_summary_highlight_suffix", " (our method)")
                    or "")
    m_names = _summary_names(plot_cfg, "stability_summary_method_labels", rows)
    m_shown = {m: m_names[m] + (hl_suffix if m in hl else "") for m in rows}
    c_names = _summary_names(plot_cfg, "stability_summary_dataset_labels", cols)
    if prettify_cols and plot_cfg.get("stability_summary_prettify_names", True):
        c_names = {c: (n if n != c else
                       re.sub(r"\s+", " ", str(c).replace("_", " ")).strip().capitalize())
                   for c, n in c_names.items()}

    phrase = phrase or _summary_value_phrase(plot_cfg)
    scale_note = "" if (native or raw) else " (Fisher z)"
    nr, nc = len(rows), len(cols)

    # ---- colour scale ---------------------------------------------------------
    finite = V.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if scale == "sequential":
        if vrange is not None:
            v_lo, v_hi = float(vrange[0]), float(vrange[1])
        elif finite.size:
            v_lo = np.floor(finite.min() * 20 + 1e-9) / 20      # out to 0.05
            v_hi = np.ceil(finite.max() * 20 - 1e-9) / 20
            if v_hi - v_lo < 0.05:
                v_hi = v_lo + 0.05
        else:
            v_lo, v_hi = 0.0, 1.0
        palette_cells = _SUMMARY_SEQUENTIAL
        key_ticks = [v_lo, v_hi]
        key_labels = [_fmt_signed(v_lo, 2), _fmt_signed(v_hi, 2)]
    else:
        vlim = plot_cfg.get("stability_summary_vlim")
        if vlim is None:
            a = np.abs(finite).max() if finite.size else 0.1
            vlim = max(0.05, np.ceil(a * 20 - 1e-9) / 20)     # round up to 0.05
        vlim = float(vlim)
        v_lo, v_hi = -vlim, vlim
        palette_cells = _SUMMARY_DIVERGING
        key_ticks = [-vlim, 0, vlim]
        key_labels = [_fmt_signed(-vlim, 1), "0", "+" + _fmt_signed(vlim, 1)]
    cmap = ListedColormap(palette_cells)
    cmap.set_bad("#f4f4f4")
    bounds = np.linspace(v_lo, v_hi, len(palette_cells) + 1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    w_lo, w_hi = direction_words

    # ====================== figure 1: heatmap + mean ± SD ======================
    if plot_cfg.get("stability_summary_heatmap", True):
        cell_w, cell_h = 0.78, 0.44
        w_heat = cell_w * nc
        w_cnt, w_for = (1.25 if count_col else 0.25), 3.0
        w_lab = 0.085 * max(len(s) for s in m_shown.values()) + 0.3
        fig_w = w_lab + w_heat + w_cnt + w_for + 0.6
        lab_in = 0.075 * max(len(s) for s in c_names.values()) * 0.6
        dendro_in = 0.75 if col_link is not None else 0.0
        # Clustered: the dendrogram sits on the heatmap and the column names move
        # underneath, as in a clustermap. Otherwise the names stay on top.
        top_in = (dendro_in + 0.75) if col_link is not None else (lab_in + 0.7)
        bot_extra = (lab_in + 0.45) if col_link is not None else 0.0
        fig_h = cell_h * nr + top_in + 1.55 + bot_extra
        fig = plt.figure(figsize=tuple(plot_cfg.get("stability_summary_heatmap_figsize")
                                       or (fig_w, fig_h)), dpi=dpi)
        fig.patch.set_facecolor("white")
        W, H = fig.get_size_inches()
        left = w_lab / W
        body_h = cell_h * nr / H
        bottom = (1.25 + bot_extra) / H
        x0 = left
        ax_h = fig.add_axes([x0, bottom, w_heat / W, body_h])
        ax_c = fig.add_axes([x0 + w_heat / W, bottom, w_cnt / W, body_h], sharey=ax_h)
        ax_f = fig.add_axes([x0 + (w_heat + w_cnt + 0.15) / W, bottom,
                             (w_for - 0.15) / W, body_h], sharey=ax_h)
        cax = fig.add_axes([x0, bottom - (0.78 + 0.55 * bot_extra) / H, min(w_heat, 2.6) / W,
                            0.16 / H])

        Z = V[rows].to_numpy(dtype=float).T            # rows = methods
        ax_h.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), np.ma.masked_invalid(Z),
                        cmap=cmap, norm=norm, edgecolors="white", linewidth=1.2)
        for i in range(nr):
            for j in range(nc):
                v = Z[i, j]
                dark = (scale == "sequential" and np.isfinite(v)
                        and norm(v) >= cmap.N - 3)
                ax_h.text(j + 0.5, i + 0.5, _fmt_signed(v), ha="center", va="center",
                          fontsize=10, color=("white" if dark else "#222222"))
        ax_h.set_xlim(0, nc)
        ax_h.set_ylim(nr, 0)                            # first row at the top
        ax_h.set_yticks(np.arange(nr) + 0.5)
        ax_h.set_yticklabels([m_shown[m] for m in rows], fontsize=11)
        for t, m in zip(ax_h.get_yticklabels(), rows):
            if m in hl:
                t.set_fontweight("bold")
        ax_h.set_xticks(np.arange(nc) + 0.5)
        if col_link is None:
            ax_h.xaxis.tick_top()
            ax_h.set_xticklabels([c_names[c] for c in cols], rotation=35, ha="left",
                                 rotation_mode="anchor", fontsize=10)
        else:
            ax_h.set_xticklabels([c_names[c] for c in cols], rotation=35, ha="right",
                                 rotation_mode="anchor", fontsize=10)
            ax_d = fig.add_axes([x0, bottom + body_h + 0.06 / H, w_heat / W,
                                 dendro_in / H])
            from scipy.cluster import hierarchy as sch
            sch.dendrogram(col_link, ax=ax_d, no_labels=True, color_threshold=0,
                           above_threshold_color="#8a8a8a")
            ax_d.set_xlim(0, 10 * nc)
            ax_d.axis("off")
            for ln in ax_d.collections:
                ln.set_linewidth(1.1)
        ax_h.tick_params(length=0, pad=4)
        for s in ax_h.spines.values():
            s.set_visible(False)
        ax_h.set_xlabel(f"{cell_stat} {phrase}{scale_note} per {unit_singular}",
                        fontsize=10.5, labelpad=8)

        # count column
        ax_c.set_xlim(0, 1)
        for i, m in (enumerate(rows) if count_col else []):
            full = int(n_pos[m]) == int(n_obs[m]) and int(n_obs[m]) > 0
            ax_c.text(0.5, i + 0.5, f"{int(n_pos[m])}/{int(n_obs[m])}", ha="center",
                      va="center", fontsize=11,
                      fontweight=("bold" if full else "normal"), color="#222222")
        ax_c.axis("off")
        if count_col:
            ax_c.text(0.5, -0.25, f"{unit_plural.capitalize()}\nwith score > 0",
                      ha="center", va="bottom", fontsize=10, transform=ax_c.transData,
                      linespacing=1.15)

        # mean ± SD
        for i, m in enumerate(rows):
            if not np.isfinite(mean[m]):
                continue
            col = palette.get(m, "#888888")
            big = m in hl
            s = sd[m] if np.isfinite(sd[m]) else 0.0
            ax_f.plot([mean[m] - s, mean[m] + s], [i + 0.5, i + 0.5], color=col,
                      lw=(2.4 if big else 1.8), solid_capstyle="round", zorder=2)
            ax_f.plot([mean[m]], [i + 0.5], "o", color=col, ms=(10 if big else 7.5),
                      markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        lo = float(np.nanmin((mean - sd.fillna(0)).to_numpy()))
        hi = float(np.nanmax((mean + sd.fillna(0)).to_numpy()))
        if zero_ref:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.08 * (hi - lo or 1.0)
        ax_f.set_xlim(lo - pad, hi + pad)
        if zero_ref:
            ax_f.axvline(0, color="#888888", ls=(0, (3, 3)), lw=1.0, zorder=1)
        ax_f.tick_params(axis="y", left=False, labelleft=False)
        ax_f.tick_params(axis="x", labelsize=9, colors="#555555")
        for k in ("top", "right", "left"):
            ax_f.spines[k].set_visible(False)
        ax_f.spines["bottom"].set_color("#999999")
        ax_f.grid(False)
        ax_f.set_title(f"Mean ± SD across {unit_plural}", fontsize=10.5,
                       pad=10, color="#222222")
        _below = matplotlib.transforms.offset_copy(ax_f.transAxes, fig=fig, y=-24,
                                                    units="points")
        ax_f.text(0.0, 0.0, f"← {w_lo}", transform=_below, ha="left",
                  va="top", fontsize=9.5, color="#555555")
        ax_f.text(1.0, 0.0, f"{w_hi} →", transform=_below, ha="right",
                  va="top", fontsize=9.5, color="#555555")
        ax_f.set_facecolor("none")

        # discrete colour key
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="horizontal", ticks=key_ticks,
                          drawedges=False)
        cb.ax.set_xticklabels(key_labels, fontsize=9, color="#555555")
        cb.outline.set_visible(False)
        cb.ax.minorticks_off()
        cb.ax.tick_params(which="both", length=0, pad=3)
        cax.text(0.0, -1.9, f"← {w_lo}", transform=cax.transAxes, ha="left",
                 va="top", fontsize=9, color="#555555")
        cax.text(1.0, -1.9, f"{w_hi} →", transform=cax.transAxes, ha="right",
                 va="top", fontsize=9, color="#555555")

        if heat_title:
            fig.suptitle(heat_title, x=left, ha="left", y=0.995, va="top",
                         fontsize=12.5, fontweight="bold")
        _save_figure(heat_base, dpi)

    # ======================= figure 2: mean vs SD scatter ======================
    if plot_cfg.get("stability_summary_scatter", True):
        ok = [m for m in rows if np.isfinite(mean[m]) and np.isfinite(sd[m])]
        if not ok:
            return
        fig, ax = plt.subplots(figsize=tuple(plot_cfg.get("stability_summary_scatter_figsize")
                                             or (7.4, 5.6)), dpi=dpi)
        fig.patch.set_facecolor("white")
        xs = np.array([sd[m] for m in ok])
        ys = np.array([mean[m] for m in ok])
        for m, xv, yv in zip(ok, xs, ys):
            big = m in hl
            ax.plot([xv], [yv], "o", color=palette.get(m, "#888888"),
                    ms=(13 if big else 10), markeredgecolor="white",
                    markeredgewidth=1.0, zorder=3)
        x_hi = float(xs.max()) * 1.18 + 1e-6
        y_lo, y_hi = float(ys.min()), float(ys.max())
        if zero_ref:
            y_lo, y_hi = min(y_lo, 0.0), max(y_hi, 0.0)
        y_pad = 0.14 * (y_hi - y_lo or 0.1)
        ax.set_xlim(0, x_hi)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        if zero_ref:
            ax.axhline(0, color="#888888", ls=(0, (4, 3)), lw=1.0, zorder=1)
            ax.text(x_hi, 0, "  below: transitions point the wrong way on average"
                    if VALUE_LABEL.strip().lower() in ("cbdir", "alignment") else "",
                    ha="right", va="bottom", fontsize=8, color="#888888")
        _place_labels(ax, fig, list(zip(xs, ys)), [m_shown[m] for m in ok],
                      [dict(fontsize=(11.5 if m in hl else 10.5),
                            fontweight=("bold" if m in hl else "normal"),
                            color="#222222") for m in ok])
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        for k in ("left", "bottom"):
            ax.spines[k].set_color("#999999")
        ax.grid(False)
        ax.tick_params(labelsize=9.5, colors="#555555")
        ax.set_ylabel(f"Mean {phrase}{scale_note}\nacross {unit_plural}  "
                      f"(higher = {w_hi} →)", fontsize=10.5, labelpad=8)
        ax.set_xlabel(f"SD of {phrase.split(' (')[0]} across {unit_plural}  "
                      f"(consistency)", fontsize=10.5, labelpad=8)
        ax.text(0.0, -0.15, "← more consistent", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#555555")
        ax.text(1.0, -0.15, "less consistent →", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555555")
        ax.annotate(best_note or "best: high and consistent", xy=(0.02, 0.98),
                    xycoords="axes fraction", ha="left", va="top", fontsize=9,
                    color="#2f7d32", fontstyle="italic")
        if scatter_title:
            ax.set_title(scatter_title, fontsize=12.5, fontweight="bold", loc="left",
                         pad=12)
        _save_figure(scatter_base, dpi)


# ---------------------------------------------------------------------------
# ICCoh stability: the summary heatmap and mean-vs-SD scatter for coherence
# ---------------------------------------------------------------------------

def build_iccoh_levels(long_df, method_order, plot_cfg):
    """(dataset, cluster) x method matrix of in-cluster coherence.

    ICCoh is scored per cell within its SOURCE cluster, and a source cell is
    re-listed once per edge leaving that cluster, so cells are de-duplicated
    before collapsing: a fork such as Pre-endocrine -> {Alpha, Beta, Delta,
    Epsilon} must not count its cells four times. Same two-stage collapse and
    the same aggregator as the CBDir-vs-ICCoh control, so the dataset values on
    the ICCoh stability figures are the x-values of that figure's points.

    Returns (levels, label); levels is empty when there is nothing to plot.
    """
    col = str(plot_cfg.get("iccoh_column", "iccoh") or "iccoh")
    if long_df is None or long_df.empty or col not in long_df.columns:
        return pd.DataFrame(), ""
    label, prov, mixed = iccoh_provenance(long_df, col)
    if mixed and not plot_cfg.get("iccoh_allow_mixed", False):
        print(f"  Skipping ICCoh stability: the long tables carry more than one kind "
              f"of '{col}' (see the CBDir-vs-ICCoh warning); set "
              f"iccoh_allow_mixed: true to plot anyway.")
        return pd.DataFrame(), label
    d = long_df.dropna(subset=[col])
    d = d[d["method"].isin(method_order)]
    if d.empty:
        print("  Skipping ICCoh stability: no rows carry an ICCoh value")
        return pd.DataFrame(), label
    if "source" not in d.columns:
        d = d.assign(source=d["edge"].astype(str).str.split(" -> ").str[0])
    if "cell_barcode" in d.columns:
        d = d.drop_duplicates(["dataset", "method", "source", "cell_barcode"])
    how = str(plot_cfg.get("iccoh_stability_aggregator")
              or plot_cfg.get("iccoh_aggregator", "median")).lower()
    fn = "median" if how.startswith("med") else "mean"
    levels = (d.groupby(["dataset", "source", "method"])[col].agg(fn)
                .unstack("method"))
    levels.index = levels.index.set_names(["dataset", "cluster"])
    levels = levels[[m for m in method_order if m in levels.columns]]
    return levels, label


def plot_iccoh_stability(long_df, method_order, plot_cfg, suffix, dataset_order=None):
    """ICCoh versions of the stability heatmap and mean-vs-SD scatter.

    Drawn with a one-hue scale and no zero rule: ICCoh is a mean cosine between
    a cell's velocity and its cluster neighbours', so 0 is not a boundary the
    way CBDir = 0 is, and higher means smoother, not more correct. The scatter
    carries that caveat in its corner note.
    """
    levels, label = build_iccoh_levels(long_df, method_order, plot_cfg)
    if levels.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    how = str(plot_cfg.get("iccoh_stability_aggregator")
              or plot_cfg.get("iccoh_aggregator", "median")).lower()
    fn = "median" if how.startswith("med") else "mean"
    ds_vals = levels.groupby(level="dataset").agg(fn)
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    phrase = str(plot_cfg.get("iccoh_stability_value_label")
                 or "in-cluster coherence (ICCoh)")
    vr = plot_cfg.get("iccoh_stability_vrange")
    common = dict(raw=True, phrase=phrase, scale="sequential", count_col=False,
                  zero_ref=False, direction_words=("less coherent", "more coherent"),
                  best_note="top left: high and consistent coherence\n"
                            "(smoothness only; ICCoh is blind to direction)",
                  vrange=vr, cell_stat=fn.capitalize())
    order = [m for m in method_order if m in ds_vals.columns]
    print(f"  covariate: '{plot_cfg.get('iccoh_column', 'iccoh')}' = {label}; "
          f"{fn} per cluster, then {fn} over a dataset's clusters")

    # CSV: the plotted values and each method's summary, so the figure can be
    # quoted without re-deriving it.
    os.makedirs(os.path.dirname(os.path.abspath(save_base)) or ".", exist_ok=True)
    rows = []
    for m in order:
        v = ds_vals[m].dropna()
        rows.append(dict(method=m, n_datasets=int(v.size),
                         mean_over_datasets=float(v.mean()) if v.size else np.nan,
                         sd_over_datasets=float(v.std(ddof=1)) if v.size > 1 else np.nan,
                         worst_dataset=float(v.min()) if v.size else np.nan,
                         worst_dataset_name=str(v.idxmin()) if v.size else "",
                         n_clusters=int(levels[m].notna().sum()),
                         aggregator=fn, iccoh_label=label))
    pd.DataFrame(rows).to_csv(f"{save_base}_iccoh_stability{suffix}.csv", index=False)
    ds_vals.to_csv(f"{save_base}_iccoh_stability_values{suffix}.csv")
    print(f"  Saved: {save_base}_iccoh_stability{suffix}.csv")

    _draw_stability_summary(
        ds_vals, order, plot_cfg, palette=palette,
        heat_base=f"{save_base}_iccoh_stability_heatmap{suffix}",
        scatter_base=f"{save_base}_iccoh_stability_mean_vs_sd{suffix}",
        unit_singular="dataset", unit_plural="datasets", col_order=dataset_order,
        heat_title=plot_cfg.get("iccoh_stability_heatmap_title"),
        scatter_title=plot_cfg.get("iccoh_stability_scatter_title"), **common)

    if not plot_cfg.get("iccoh_stability_per_dataset_plot",
                        plot_cfg.get("stability_per_dataset_summary_plot", True)):
        return
    sub = plot_cfg.get("stability_per_dataset_subdir", "stability_per_dataset")
    out_dir = os.path.dirname(os.path.abspath(save_base))
    if sub:
        out_dir = os.path.join(out_dir, str(sub))
    stem = os.path.basename(save_base)
    names = [d for d in (dataset_order or []) if d in ds_vals.index]
    names += [d for d in ds_vals.index if d not in names]
    for d_name in names:
        vals = levels.xs(d_name, level="dataset").dropna(how="all")
        if vals.shape[0] < 2:
            continue                        # one cluster has no spread to show
        slug = _slug(d_name)
        _draw_stability_summary(
            vals, order, plot_cfg, palette=palette,
            heat_base=os.path.join(out_dir, f"{stem}_iccoh_stability_heatmap_{slug}{suffix}"),
            scatter_base=os.path.join(out_dir,
                                      f"{stem}_iccoh_stability_mean_vs_sd_{slug}{suffix}"),
            unit_singular="cluster", unit_plural="clusters",
            heat_title=d_name, scatter_title=d_name, prettify_cols=False, **common)


# ---------------------------------------------------------------------------
# Linear mixed models
# ---------------------------------------------------------------------------

def _offset_stats(d, edge_labels, cluster_unit="edge"):
    """Offset of a method vs the reference from per-cell paired differences.

    Two units of replication, both closed-form (no iterative optimizer):

    cell  - every scored cell is a replicate: one-sample t on all differences,
            df = n_cells - 1. Equivalent to the random-intercept mixed model on
            the level scale, because the cell random effect cancels exactly in a
            within-cell contrast (verified to ~1e-9 on the SE). It does NOT
            account for cells within a transition being correlated, so it is
            anticonservative — reported, but not the default.
    edge  - the transition is the replicate: average the differences within each
            edge, then a one-sample t across edge means, df = n_edges - 1. This
            is what respects the clustering.

    Returns a dict with both, plus which one is primary.
    """
    d = np.asarray(d, dtype=np.float64)
    edge_labels = np.asarray(edge_labels, dtype=object)
    out = {}

    n = d.size
    mean_all = float(np.mean(d)) if n else np.nan
    if n > 1:
        se_c = float(stats.sem(d))
        t_c = mean_all / se_c if se_c > 0 else np.nan
        df_c = n - 1
        p_c = float(2 * stats.t.sf(abs(t_c), df_c)) if np.isfinite(t_c) else np.nan
    else:
        se_c = t_c = p_c = np.nan
        df_c = 0
    out.update(offset_cell=mean_all, se_cell=se_c, t_cell=t_c, df_cell=df_c, p_cell=p_c)

    # Probability of superiority: does the method beat the reference for this
    # cell? Aggregated per edge so the clustering is respected, exactly as the
    # magnitude test is. Ties count as half a win.
    win = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))

    uniq = pd.unique(edge_labels)
    means = np.array([np.mean(d[edge_labels == e]) for e in uniq], dtype=np.float64)
    wins = np.array([np.mean(win[edge_labels == e]) for e in uniq], dtype=np.float64)
    out["p_win_cell"] = float(np.mean(win)) if win.size else np.nan
    out["edge_means"] = means
    out["edge_wins"] = wins
    out["edge_labels"] = list(uniq)
    if wins.size > 1:
        mw, sw = float(np.mean(wins)), float(stats.sem(wins))
        tw = (mw - 0.5) / sw if sw > 0 else np.nan
        pw = float(2 * stats.t.sf(abs(tw), wins.size - 1)) if np.isfinite(tw) else np.nan
        crit = float(stats.t.ppf(0.975, wins.size - 1))
        out.update(p_win=mw, p_win_se=sw, p_win_p=pw,
                   p_win_ci_low=mw - crit * sw, p_win_ci_high=mw + crit * sw,
                   odds_ratio=float(mw / (1 - mw)) if 0 < mw < 1 else np.nan)
    else:
        out.update(p_win=float(wins[0]) if wins.size else np.nan, p_win_se=np.nan,
                   p_win_p=np.nan, p_win_ci_low=np.nan, p_win_ci_high=np.nan,
                   odds_ratio=np.nan)
    k = means.size
    out["n_edges_used"] = int(k)
    if k > 1:
        mean_e = float(np.mean(means))
        se_e = float(stats.sem(means))
        t_e = mean_e / se_e if se_e > 0 else np.nan
        df_e = k - 1
        p_e = float(2 * stats.t.sf(abs(t_e), df_e)) if np.isfinite(t_e) else np.nan
        crit = float(stats.t.ppf(0.975, df_e))
        ci_e = (mean_e - crit * se_e, mean_e + crit * se_e)
    else:
        mean_e = float(means[0]) if k else np.nan
        se_e = t_e = p_e = np.nan
        df_e = 0
        ci_e = (np.nan, np.nan)
    out.update(offset_edge=mean_e, se_edge=se_e, t_edge=t_e, df_edge=df_e, p_edge=p_e,
               ci_edge_low=ci_e[0], ci_edge_high=ci_e[1])

    if cluster_unit == "cell" or k <= 1:
        if k <= 1 and cluster_unit == "edge":
            out["notes"] = (f"only {k} edge(s): edge-level test impossible, "
                            f"fell back to the cell-level test (anticonservative)")
        crit = float(stats.t.ppf(0.975, df_c)) if df_c > 0 else np.nan
        out.update(offset_z=mean_all, se_z=se_c, t_stat=t_c, df=df_c, p=p_c,
                   ci_z_low=mean_all - crit * se_c if np.isfinite(crit) else np.nan,
                   ci_z_high=mean_all + crit * se_c if np.isfinite(crit) else np.nan,
                   unit_used="cell")
    else:
        out.update(offset_z=mean_e, se_z=se_e, t_stat=t_e, df=df_e, p=p_e,
                   ci_z_low=ci_e[0], ci_z_high=ci_e[1], unit_used="edge")
    return out


def _fit_pair(pair_long, transform, eps, edge_effect, reml):
    """Fit one {reference, method} pair. pair_long has obs_id, edge_tok, grp, cbdir.

    grp is the internal two-level factor: 'ref' (baseline) and 'alt'. Method and
    edge names are tokenised before fitting because real names contain hyphens,
    spaces and '->', which make patsy term names awkward to address.
    """
    import statsmodels.formula.api as smf

    z, n_clipped = _forward_transform(pair_long["cbdir"].to_numpy(), transform, eps)
    d = pair_long.copy()
    d["value"] = z
    d["grp"] = pd.Categorical(d["grp"], categories=["ref", "alt"])

    n_edges = d["edge_tok"].nunique()
    term = "C(grp)[T.alt]"
    notes = []

    if edge_effect == "random":
        res = smf.mixedlm("value ~ C(grp)", d, groups=d["edge_tok"],
                          vc_formula={"obs": "0 + C(obs_id)"}).fit(reml=reml)
        sigma_edge = float(np.sqrt(res.cov_re.iloc[0, 0])) if res.cov_re.shape[0] else np.nan
        sigma_cell = float(np.sqrt(res.vcomp[0])) if len(res.vcomp) else np.nan
    else:
        formula = "value ~ C(grp) + C(edge_tok)" if n_edges > 1 else "value ~ C(grp)"
        if n_edges <= 1:
            notes.append("single edge: C(edge) dropped")
        res = smf.mixedlm(formula, d, groups=d["obs_id"]).fit(reml=reml)
        sigma_edge = np.nan
        sigma_cell = float(np.sqrt(res.cov_re.iloc[0, 0])) if res.cov_re.shape[0] else np.nan

    sigma_resid = float(np.sqrt(res.scale))
    offset_z = float(res.params[term])
    se_z = float(res.bse[term])
    p = float(res.pvalues[term])
    ci = res.conf_int()
    ci_lo, ci_hi = float(ci.loc[term, 0]), float(ci.loc[term, 1])

    z_ref_mean = float(np.mean(d.loc[d["grp"] == "ref", "value"]))
    offset_native = _native_offset(z_ref_mean, offset_z, transform)

    denom = (sigma_cell ** 2 + sigma_resid ** 2) if np.isfinite(sigma_cell) else np.nan
    icc_cell = float(sigma_cell ** 2 / denom) if np.isfinite(denom) and denom > 0 else np.nan

    singular = bool(np.isfinite(sigma_cell) and sigma_cell < 1e-6)
    if singular:
        notes.append("singular: cell variance ~ 0")
    if not getattr(res, "converged", True):
        notes.append("did not converge")

    return dict(
        offset_z=offset_z, se_z=se_z, z_stat=float(res.tvalues[term]), p=p,
        ci_z_low=ci_lo, ci_z_high=ci_hi, offset_native=offset_native,
        ci_native_low=_native_offset(z_ref_mean, ci_lo, transform),
        ci_native_high=_native_offset(z_ref_mean, ci_hi, transform),
        sigma_edge=sigma_edge, sigma_cell=sigma_cell, sigma_resid=sigma_resid,
        icc_cell=icc_cell, n_clipped=n_clipped,
        converged=bool(getattr(res, "converged", True)), singular=singular,
        notes="; ".join(notes),
    ), res, d


def _fit_slope(pair_wide, ref, m, transform, eps, reml):
    """model: slope — value_m ~ value_ref with (1|edge). Tests slope = 1."""
    import statsmodels.formula.api as smf
    d = pair_wide.copy()
    d["y"], _ = _forward_transform(d[m].to_numpy(), transform, eps)
    d["x"], _ = _forward_transform(d[ref].to_numpy(), transform, eps)
    res = smf.mixedlm("y ~ x", d, groups=d["edge_tok"]).fit(reml=reml)
    slope, slope_se = float(res.params["x"]), float(res.bse["x"])
    p_slope = float(2 * stats.norm.sf(abs((slope - 1.0) / slope_se))) if slope_se > 0 else np.nan
    return dict(slope=slope, slope_se=slope_se, p_slope_eq_1=p_slope,
                intercept=float(res.params["Intercept"]))


def fit_lmm(long_df, method_order, dataset_order, plot_cfg):
    """Fit the per-method reference-offset models. Returns (results_df, residuals_df)."""
    try:
        import statsmodels.formula.api as smf  # noqa: F401
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        print("WARNING: statsmodels is not installed; skipping linear models. "
              "Install it with `pip install statsmodels`.")
        return pd.DataFrame(), pd.DataFrame()

    ref = plot_cfg.get("reference_method")
    if not ref or ref not in method_order:
        print(f"WARNING: reference_method '{ref}' not among methods {method_order}; skipping LMM.")
        return pd.DataFrame(), pd.DataFrame()

    model_kind = str(plot_cfg.get("model", "per_method")).lower()
    cluster_unit = str(plot_cfg.get("cluster_unit", "edge")).lower()
    estimator = str(plot_cfg.get("estimator", "closed_form")).lower()
    if cluster_unit not in ("edge", "cell"):
        raise ValueError(f"cluster_unit must be edge | cell, got '{cluster_unit}'")
    transform = str(plot_cfg.get("transform", "atanh")).lower()
    eps = _cfg_num(plot_cfg, "transform_eps", 1e-6)
    edge_effect = str(plot_cfg.get("edge_effect", "fixed")).lower()
    reml = bool(plot_cfg.get("reml", True))
    p_adjust = str(plot_cfg.get("p_adjust", "fdr_bh")).lower()
    stat_test = str(plot_cfg.get("stat_test", "wilcoxon")).lower()

    print(f"\n=== Offset models (cluster_unit={cluster_unit}, estimator={estimator}, "
          f"transform={transform}, reference='{ref}') ===")
    if cluster_unit == "cell":
        print("  WARNING: cluster_unit='cell' treats every cell as an independent "
              "replicate. Cells within a transition are correlated, so p-values are "
              "anticonservative. Use cluster_unit='edge' for inference.")

    rows, resid_rows, per_edge_rows = [], [], []

    for d_name in dataset_order:
        dsub = long_df[long_df["dataset"] == d_name]
        wide = dsub.pivot_table(index="obs_id", columns="method", values="cbdir", aggfunc="first")
        edge_of = dsub.drop_duplicates("obs_id").set_index("obs_id")["edge"]
        edge_tokens = {e: f"e{i}" for i, e in enumerate(sorted(pd.unique(dsub["edge"])))}

        if ref not in wide.columns:
            print(f"  [{d_name}] reference '{ref}' absent; skipping dataset.")
            continue

        dataset_rows = []
        for m in method_order:
            if m == ref or m not in wide.columns:
                continue
            pair = wide[[ref, m]].dropna()
            n_dropped = int(wide[[ref, m]].notna().any(axis=1).sum() - len(pair))
            if len(pair) < 10:
                print(f"  [{d_name}] {m}: only {len(pair)} complete pairs; skipping.")
                continue

            pair = pair.copy()
            pair["edge"] = edge_of.reindex(pair.index)
            pair["edge_tok"] = pair["edge"].map(edge_tokens)

            pair_long = pd.concat([
                pd.DataFrame(dict(obs_id=pair.index, edge_tok=pair["edge_tok"].to_numpy(),
                                  grp="ref", cbdir=pair[ref].to_numpy())),
                pd.DataFrame(dict(obs_id=pair.index, edge_tok=pair["edge_tok"].to_numpy(),
                                  grp="alt", cbdir=pair[m].to_numpy())),
            ], ignore_index=True)

            z_alt, n_clip_a = _forward_transform(pair[m].to_numpy(), transform, eps)
            z_ref, n_clip_r = _forward_transform(pair[ref].to_numpy(), transform, eps)
            d_vals = z_alt - z_ref
            z_ref_mean = float(np.mean(z_ref))

            rec = dict(dataset=d_name, method=m, reference_method=ref,
                       estimator=estimator, cluster_unit=cluster_unit,
                       transform=transform,
                       n_obs=2 * len(pair), n_cells=len(pair),
                       n_edges=int(pair["edge"].nunique()),
                       n_dropped_incomplete=n_dropped,
                       n_clipped=int(n_clip_a + n_clip_r))

            st = _offset_stats(d_vals, pair["edge"].to_numpy(), cluster_unit)
            for e_lab, e_mean, e_win in zip(st["edge_labels"], st["edge_means"], st["edge_wins"]):
                per_edge_rows.append(dict(dataset=d_name, method=m, edge=e_lab,
                                          mean_diff_z=float(e_mean), win_rate=float(e_win),
                                          z_ref_mean=z_ref_mean))
            st = {k: v for k, v in st.items()
                  if k not in ("edge_means", "edge_wins", "edge_labels")}
            rec.update(st)
            rec["offset_native"] = _native_offset(z_ref_mean, st["offset_z"], transform)
            rec["ci_native_low"] = _native_offset(z_ref_mean, st["ci_z_low"], transform) \
                if np.isfinite(st["ci_z_low"]) else np.nan
            rec["ci_native_high"] = _native_offset(z_ref_mean, st["ci_z_high"], transform) \
                if np.isfinite(st["ci_z_high"]) else np.nan
            rec["offset_native_cell"] = _native_offset(z_ref_mean, st["offset_cell"], transform)

            if estimator == "mixedlm":
                try:
                    pair_long = pd.concat([
                        pd.DataFrame(dict(obs_id=pair.index, edge_tok=pair["edge_tok"].to_numpy(),
                                          grp="ref", cbdir=pair[ref].to_numpy())),
                        pd.DataFrame(dict(obs_id=pair.index, edge_tok=pair["edge_tok"].to_numpy(),
                                          grp="alt", cbdir=pair[m].to_numpy())),
                    ], ignore_index=True)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit, res, _ = _fit_pair(pair_long, transform, eps, edge_effect, reml)
                    names = sorted({w.category.__name__ for w in caught})
                    rec.update({f"mixedlm_{k}": v for k, v in fit.items()
                                if k in ("offset_z", "se_z", "p", "sigma_cell",
                                         "sigma_resid", "icc_cell", "converged")})
                    if names:
                        rec["mixedlm_warnings"] = ";".join(names)
                    resid_rows.append(pd.DataFrame(dict(
                        dataset=d_name, method=m,
                        fitted=np.asarray(res.fittedvalues, dtype=float),
                        resid=np.asarray(res.resid, dtype=float))))
                except Exception as e:
                    print(f"  [{d_name}] {m}: optional MixedLM failed: {type(e).__name__}: {e}")
            else:
                resid_rows.append(pd.DataFrame(dict(
                    dataset=d_name, method=m,
                    fitted=np.full(d_vals.size, st["offset_z"]),
                    resid=d_vals - st["offset_z"])))

            # Nonparametric cross-check on the native scale.
            p_np, n_pairs = _paired_pvalue(pair[ref], pair[m], stat_test)
            rec["wilcoxon_p"] = p_np
            rec["median_diff"] = float(np.median(pair[m].to_numpy() - pair[ref].to_numpy()))
            dataset_rows.append(rec)

            print(f"  [{d_name}] {m:<20s} offset_z={rec.get('offset_z', np.nan):+.4f} "
                  f"(native {rec.get('offset_native', np.nan):+.4f})  "
                  f"p={rec.get('p', np.nan):.3g} [{rec.get('unit_used')}, "
                  f"df={rec.get('df')}]  n_cells={len(pair)}, "
                  f"n_edges={rec.get('n_edges_used')}"
                  + (f"  ({rec['notes']})" if rec.get("notes") else ""))

        if dataset_rows and p_adjust != "none":
            pvals = [r.get("p", np.nan) for r in dataset_rows]
            ok = [i for i, v in enumerate(pvals) if v is not None and np.isfinite(v)]
            adj = [np.nan] * len(pvals)
            if ok:
                _, padj, _, _ = multipletests([pvals[i] for i in ok], method=p_adjust)
                for i, v in zip(ok, padj):
                    adj[i] = float(v)
            for r, v in zip(dataset_rows, adj):
                r["p_adj"] = v
        else:
            for r in dataset_rows:
                r["p_adj"] = r.get("p", np.nan)

        rows.extend(dataset_rows)

    results = pd.DataFrame(rows)
    residuals = pd.concat(resid_rows, ignore_index=True) if resid_rows else pd.DataFrame()
    per_edge = pd.DataFrame(per_edge_rows)
    return results, residuals, per_edge


def _one_sample_t(x, mu=0.0):
    """mean, se, t, df, p, 95% CI for a one-sample t-test."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return dict(mean=float(x[0]) if n else np.nan, se=np.nan, t=np.nan,
                    df=max(n - 1, 0), p=np.nan, ci_low=np.nan, ci_high=np.nan, n=n)
    m, se = float(np.mean(x)), float(stats.sem(x))
    t = (m - mu) / se if se > 0 else np.nan
    df = n - 1
    p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else np.nan
    crit = float(stats.t.ppf(0.975, df))
    return dict(mean=m, se=se, t=t, df=df, p=p,
                ci_low=m - crit * se, ci_high=m + crit * se, n=n)


def _dataset_fixed_t(values, ds_labels, mu=0.0):
    """Dataset as a FIXED effect, edge-within-dataset as the error term.

    The estimate is the unweighted mean of the per-dataset means; the SE comes
    from the pooled within-dataset between-edge variance, on n_edges - n_datasets
    degrees of freedom. Equivalent to OLS on the edge means with dataset dummies,
    read off as a marginal mean.

    This conditions on the datasets at hand: the null is "the average offset over
    THESE datasets is zero". It does not propagate dataset-to-dataset
    heterogeneity into the SE, which is exactly what makes it more powerful and
    also what narrows the claim.
    """
    d = pd.DataFrame(dict(v=np.asarray(values, dtype=np.float64),
                          ds=np.asarray(ds_labels, dtype=object)))
    d = d[np.isfinite(d["v"])]
    k = d["ds"].nunique()
    n = len(d)
    df = n - k
    if k < 1 or df < 1:
        return dict(mean=float(d["v"].mean()) if n else np.nan, se=np.nan, t=np.nan,
                    df=max(df, 0), p=np.nan, ci_low=np.nan, ci_high=np.nan, n=n,
                    note="need >1 edge in at least one dataset")
    dm = d.groupby("ds")["v"].mean()
    ssw = float(sum(((g["v"] - g["v"].mean()) ** 2).sum() for _, g in d.groupby("ds")))
    s2 = ssw / df
    nd = d.groupby("ds").size().to_numpy(dtype=np.float64)
    se = float(np.sqrt(s2 / (k ** 2) * np.sum(1.0 / nd)))
    m = float(dm.mean())
    t = (m - mu) / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else np.nan
    crit = float(stats.t.ppf(0.975, df))
    return dict(mean=m, se=se, t=t, df=df, p=p,
                ci_low=m - crit * se, ci_high=m + crit * se, n=n, note="")


def _signed_rank(x, mu=0.0):
    """One-sample Wilcoxon signed-rank test with a matched-pairs effect size.

    Applied to the per-edge mean differences, so a transition is one observation
    and the parametric assumptions on the differences are dropped. The effect
    size is the matched-pairs rank-biserial correlation,
    (W+ - W-) / (W+ + W-), which runs from -1 (every transition favours the
    reference) to +1 (every transition favours the method).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)] - mu
    x = x[x != 0]                      # exact ties carry no sign information
    n = x.size
    if n == 0:
        return dict(p=np.nan, n=0, rbc=np.nan, median=np.nan,
                    note="no non-zero differences")

    # The effect size is descriptive and well defined at any n, so it is always
    # reported; only the p-value is withheld when the test cannot deliver one.
    ranks = stats.rankdata(np.abs(x))
    wp, wm = float(ranks[x > 0].sum()), float(ranks[x < 0].sum())
    rbc = (wp - wm) / (wp + wm) if (wp + wm) > 0 else np.nan
    med = float(np.median(x))

    if n < 6:
        # Below 6 non-zero pairs the two-sided exact test cannot reach p < 0.05
        # for any data, so a p-value here would be misleading rather than null.
        return dict(p=np.nan, n=n, rbc=float(rbc), median=med,
                    note=f"only {n} non-zero transitions; two-sided signed-rank "
                         f"cannot reach p<0.05 at this n")
    try:
        _, p = stats.wilcoxon(x, alternative="two-sided")
    except Exception as e:
        return dict(p=np.nan, n=n, rbc=float(rbc), median=med, note=str(e))
    return dict(p=float(p), n=n, rbc=float(rbc), median=med, note="")


def _cluster_sandwich(scores, hess, groups):
    """Cluster-robust variance for a one-parameter score, clustered on `groups`.

    Score contributions are summed WITHIN each cluster before being squared, so
    the estimator counts clusters rather than observations as the independent
    units. Includes the usual finite-cluster scaling G/(G-1); the caller should
    use t with G-1 degrees of freedom, which matters a great deal when G is small
    (7 datasets) and hardly at all when it is large (30 transitions).
    """
    g = pd.Series(np.asarray(scores, dtype=np.float64)).groupby(
        pd.Series(np.asarray(groups, dtype=object))).sum().to_numpy()
    G = g.size
    if hess == 0 or not np.isfinite(hess) or G < 2:
        return np.nan, G
    meat = float(np.sum(g ** 2)) * G / (G - 1.0)
    bread = 1.0 / hess
    return float(bread * meat * bread), G


def conditional_logit_paired(d, clusters):
    """Paired (conditional) logistic: which member of the pair is the method?

    Strata are cells, one reference observation and one method observation each,
    so the conditional likelihood collapses to P = expit(beta * d_i) with
    d_i = z_method,i - z_ref,i and no intercept. Everything constant within a
    cell — dataset, transition, cell identity — cancels exactly, which is why a
    dataset fixed effect is not merely unnecessary here but inestimable.

    beta is the log-odds that the larger of the two scores belongs to the method,
    per unit of z. `clusters` is a dict of {label: array}; a cluster-robust SE is
    returned for each, because the choice of clustering level IS the assumption
    about which units are independent:

      edge    - cells are correlated inside a transition. ~30 clusters.
      dataset - additionally, transitions from one dataset share a kNN graph,
                preprocessing and biology. Clustering at the coarser level
                absorbs ALL finer dependence nested inside it, so this is the
                two-level "cells in transitions in datasets" answer. Only ~7
                clusters, so it is approximate and conservative.

    The point estimate is identical across clustering levels; only the
    uncertainty changes.
    """
    from scipy.optimize import minimize_scalar
    d = np.asarray(d, dtype=np.float64)
    keep = np.isfinite(d)
    d = d[keep]
    clusters = {k: np.asarray(v, dtype=object)[keep] for k, v in clusters.items()}
    if d.size < 10 or np.allclose(d, 0):
        out = dict(beta=np.nan, odds_ratio=np.nan, n=int(d.size),
                   note="too few or degenerate pairs")
        for k in clusters:
            out.update({f"se_{k}": np.nan, f"p_{k}": np.nan, f"n_clusters_{k}": 0,
                        f"or_per_0p1z_low_{k}": np.nan, f"or_per_0p1z_high_{k}": np.nan})
        out["or_per_0p1z"] = np.nan
        return out

    def nll(b):
        z = np.clip(b * d, -500, 500)
        return float(np.sum(np.logaddexp(0.0, -z)))

    res = minimize_scalar(nll, bounds=(-50, 50), method="bounded")
    beta = float(res.x)
    pr = 1.0 / (1.0 + np.exp(-np.clip(beta * d, -500, 500)))
    scores = d * (1.0 - pr)
    hess = float(np.sum(d ** 2 * pr * (1.0 - pr)))

    out = dict(beta=beta, odds_ratio=float(np.exp(beta)),
               or_per_0p1z=float(np.exp(0.1 * beta)), n=int(d.size), note="")
    for lab, grp in clusters.items():
        var, G = _cluster_sandwich(scores, hess, grp)
        se = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan
        if np.isfinite(se) and se > 0 and G > 1:
            crit = float(stats.t.ppf(0.975, G - 1))
            p = float(2 * stats.t.sf(abs(beta / se), G - 1))
            lo, hi = beta - crit * se, beta + crit * se
        else:
            p = lo = hi = np.nan
        out.update({f"se_{lab}": se, f"p_{lab}": p, f"n_clusters_{lab}": int(G),
                    f"or_per_0p1z_low_{lab}": float(np.exp(0.1 * lo)) if np.isfinite(lo) else np.nan,
                    f"or_per_0p1z_high_{lab}": float(np.exp(0.1 * hi)) if np.isfinite(hi) else np.nan})
    return out


def logit_unpaired(cbdir_m, cbdir_ref, ds_m, ds_ref, edge_m, edge_ref):
    """Unpaired logistic: can CBDir tell the method apart from the reference?

    logit P(method) ~ beta * z + C(dataset), fit on every cell either method
    scored, with dataset as a FIXED effect so between-dataset shifts are removed
    rather than mistaken for a method difference. SEs are cluster-robust on the
    transition. Unlike a comparison of means this uses the whole distribution, so
    two methods with equal means but different spread are still separable.
    """
    import statsmodels.api as sm
    y = np.r_[np.ones(len(cbdir_m)), np.zeros(len(cbdir_ref))]
    z = np.r_[np.asarray(cbdir_m, dtype=np.float64), np.asarray(cbdir_ref, dtype=np.float64)]
    ds = np.r_[np.asarray(ds_m, dtype=object), np.asarray(ds_ref, dtype=object)]
    cl = np.r_[np.asarray(edge_m, dtype=object), np.asarray(edge_ref, dtype=object)]
    ok = np.isfinite(z)
    y, z, ds, cl = y[ok], z[ok], ds[ok], cl[ok]
    clusters = {"edge": cl, "dataset": ds}
    if y.sum() < 10 or (len(y) - y.sum()) < 10 or len(np.unique(cl)) < 5:
        out = dict(beta=np.nan, odds_ratio=np.nan, or_per_0p1z=np.nan,
                   n=int(len(y)), note="too few cells or clusters")
        for k in clusters:
            out.update({f"se_{k}": np.nan, f"p_{k}": np.nan, f"n_clusters_{k}": 0,
                        f"or_per_0p1z_low_{k}": np.nan, f"or_per_0p1z_high_{k}": np.nan})
        return out
    X = pd.get_dummies(pd.Series(ds, name="ds"), drop_first=True).astype(float)
    X.insert(0, "z", z)
    X = sm.add_constant(X, has_constant="add")
    Xn = X.to_numpy(dtype=float)
    out = dict(n=int(len(y)), note="")
    try:
        base = sm.Logit(y, Xn).fit(disp=0, maxiter=200)
        b = float(base.params[1])
        out.update(beta=b, odds_ratio=float(np.exp(b)), or_per_0p1z=float(np.exp(0.1 * b)))
    except Exception as e:
        out.update(beta=np.nan, odds_ratio=np.nan, or_per_0p1z=np.nan,
                   note=f"{type(e).__name__}: {e}")
        for k in clusters:
            out.update({f"se_{k}": np.nan, f"p_{k}": np.nan, f"n_clusters_{k}": 0,
                        f"or_per_0p1z_low_{k}": np.nan, f"or_per_0p1z_high_{k}": np.nan})
        return out
    for lab, grp in clusters.items():
        G = int(len(np.unique(grp)))
        try:
            r = sm.Logit(y, Xn).fit(disp=0, maxiter=200, cov_type="cluster",
                                    cov_kwds={"groups": grp})
            se = float(r.bse[1])
            crit = float(stats.t.ppf(0.975, max(G - 1, 1)))
            out.update({f"se_{lab}": se,
                        f"p_{lab}": float(2 * stats.t.sf(abs(b / se), max(G - 1, 1))),
                        f"n_clusters_{lab}": G,
                        f"or_per_0p1z_low_{lab}": float(np.exp(0.1 * (b - crit * se))),
                        f"or_per_0p1z_high_{lab}": float(np.exp(0.1 * (b + crit * se)))})
        except Exception:
            out.update({f"se_{lab}": np.nan, f"p_{lab}": np.nan, f"n_clusters_{lab}": G,
                        f"or_per_0p1z_low_{lab}": np.nan, f"or_per_0p1z_high_{lab}": np.nan})
    return out


def build_per_edge_unpaired(long_df, ref, transform="atanh", eps=1e-6):
    """Per-transition offsets WITHOUT subsetting to the reference's cells.

    Each method's per-edge mean uses every cell that method scored, so nothing is
    discarded. Pairing is kept where it matters for the test — both methods still
    contribute one number per transition — but the two numbers now rest on
    different cell sets.

    That is the trade. The paired version controls cell composition exactly and
    pays for it in discarded cells (22-51% on this benchmark, and the retained
    set is defined by whichever cells the REFERENCE happened to score). The
    unpaired version uses everything and instead confounds velocity quality with
    which cells each method's kNN graph made scorable at all - a method whose
    graph qualifies only well-connected interior cells is scored on an easier
    subset. Neither is strictly better; agreement between them is the evidence
    that the choice does not matter.

    `win_rate` here is the unpaired probability of superiority, the Mann-Whitney
    AUC: the chance a random cell from this method scores above a random cell
    from the reference. That is precisely the effect size a logistic
    discriminating the two methods would report, obtained without fitting one.
    """
    rows = []
    for (d_name, e_lab), g in long_df.groupby(["dataset", "edge"], sort=False):
        ref_vals = g.loc[g["method"] == ref, "cbdir"].to_numpy()
        if ref_vals.size == 0:
            continue
        z_ref, _ = _forward_transform(ref_vals, transform, eps)
        for m, gm in g.groupby("method", sort=False):
            if m == ref:
                continue
            m_vals = gm["cbdir"].to_numpy()
            if m_vals.size == 0:
                continue
            z_m, _ = _forward_transform(m_vals, transform, eps)
            try:
                u, _ = stats.mannwhitneyu(m_vals, ref_vals, alternative="two-sided")
                auc = float(u) / (m_vals.size * ref_vals.size)
            except Exception:
                auc = np.nan
            rows.append(dict(dataset=d_name, method=m, edge=e_lab,
                             mean_diff_z=float(np.mean(z_m) - np.mean(z_ref)),
                             win_rate=auc, z_ref_mean=float(np.mean(z_ref)),
                             n_cells_method=int(m_vals.size),
                             n_cells_ref=int(ref_vals.size)))
    return pd.DataFrame(rows)


def fit_pooled(per_edge, plot_cfg):
    """Pool the per-edge differences across datasets, per method.

    Two replicates are reported for every method:
      dataset       - dataset as a RANDOM effect: average the edge differences
                      within each dataset, then test across datasets
                      (df = n_datasets - 1). The error term is between-dataset
                      heterogeneity, so the claim generalises beyond the datasets
                      at hand. Conservative; the default headline.
      dataset_fixed - dataset as a FIXED effect, edge-within-dataset as the error
                      (df = n_edges - n_datasets). Conditions on these datasets:
                      "averaged over THESE systems, is the method better?" Much
                      more power, and a correspondingly narrower claim.
      edge          - all (dataset, edge) pairs pooled, dataset ignored. Kept for
                      reference only: its error term is inflated by between-dataset
                      mean shifts, so it answers neither question cleanly.
    """
    if per_edge.empty:
        return pd.DataFrame()
    from statsmodels.stats.multitest import multipletests

    transform = str(plot_cfg.get("transform", "atanh")).lower()
    pool_unit = str(plot_cfg.get("pool_unit", "dataset")).lower()
    if pool_unit not in ("dataset", "dataset_fixed", "edge"):
        raise ValueError(f"pool_unit must be dataset | dataset_fixed | edge, got '{pool_unit}'")
    if pool_unit == "edge":
        print("  WARNING: pool_unit='edge' ignores dataset structure; its error term is "
              "inflated by between-dataset mean shifts. Prefer 'dataset' (generalising) "
              "or 'dataset_fixed' (conditional on these datasets).")
    p_adjust = str(plot_cfg.get("p_adjust", "fdr_bh")).lower()
    ref = plot_cfg.get("reference_method")

    pairing = str(plot_cfg.get("_pairing_label", "paired"))
    print(f"\n=== Pooled across datasets (headline unit = {pool_unit}, "
          f"pairing = {pairing}) ===")
    rows = []
    for m, g in per_edge.groupby("method", sort=False):
        ds_means = g.groupby("dataset")["mean_diff_z"].mean()
        ds_wins = g.groupby("dataset")["win_rate"].mean()
        a = _one_sample_t(ds_means.to_numpy())            # dataset random
        b = _one_sample_t(g["mean_diff_z"].to_numpy())    # edge, dataset ignored
        f = _dataset_fixed_t(g["mean_diff_z"], g["dataset"])          # dataset fixed
        wa = _one_sample_t(ds_wins.to_numpy(), mu=0.5)
        wb = _one_sample_t(g["win_rate"].to_numpy(), mu=0.5)
        wf = _dataset_fixed_t(g["win_rate"], g["dataset"], mu=0.5)
        # Nonparametric, transition as the observation.
        sr_e = _signed_rank(g["mean_diff_z"].to_numpy())
        sr_d = _signed_rank(ds_means.to_numpy())

        z_ref = float(np.mean(g["z_ref_mean"]))
        prim = {"dataset": a, "dataset_fixed": f, "edge": b}.get(pool_unit, a)
        wprim = {"dataset": wa, "dataset_fixed": wf, "edge": wb}.get(pool_unit, wa)
        rec = dict(
            method=m, reference_method=ref, pool_unit=pool_unit, pairing=pairing,
            n_datasets=int(g["dataset"].nunique()), n_edges=int(len(g)),
            offset_z=prim["mean"], se_z=prim["se"], t_stat=prim["t"],
            df=prim["df"], p=prim["p"], ci_z_low=prim["ci_low"], ci_z_high=prim["ci_high"],
            offset_native=_native_offset(z_ref, prim["mean"], transform),
            ci_native_low=_native_offset(z_ref, prim["ci_low"], transform),
            ci_native_high=_native_offset(z_ref, prim["ci_high"], transform),
            offset_z_dataset=a["mean"], p_dataset=a["p"], df_dataset=a["df"],
            offset_z_edge=b["mean"], p_edge=b["p"], df_edge=b["df"],
            offset_z_dsfixed=f["mean"], se_dsfixed=f["se"], p_dsfixed=f["p"],
            df_dsfixed=f["df"],
            ci_z_low_dsfixed=f["ci_low"], ci_z_high_dsfixed=f["ci_high"],
            p_win=wprim["mean"], p_win_ci_low=wprim["ci_low"],
            p_win_ci_high=wprim["ci_high"], p_win_p=wprim["p"], p_win_df=wprim["df"],
            p_win_dataset=wa["mean"], p_win_p_dataset=wa["p"],
            p_win_dsfixed=wf["mean"], p_win_p_dsfixed=wf["p"],
            p_win_edge=wb["mean"], p_win_p_edge=wb["p"],
            odds_ratio=float(wprim["mean"] / (1 - wprim["mean"]))
            if np.isfinite(wprim["mean"]) and 0 < wprim["mean"] < 1 else np.nan,
            wilcox_p_edge=sr_e["p"], wilcox_n_edge=sr_e["n"], wilcox_rbc_edge=sr_e["rbc"],
            wilcox_median_edge=sr_e["median"],
            wilcox_p_dataset=sr_d["p"], wilcox_n_dataset=sr_d["n"],
            wilcox_rbc_dataset=sr_d["rbc"],
            notes="; ".join([x for x in (f["note"], sr_e["note"]) if x]),
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    for col, adj in (("p", "p_adj"), ("p_win_p", "p_win_p_adj"),
                     ("p_dataset", "p_dataset_adj"), ("p_dsfixed", "p_dsfixed_adj"),
                     ("p_edge", "p_edge_adj"),
                     ("wilcox_p_edge", "wilcox_p_edge_adj"),
                     ("wilcox_p_dataset", "wilcox_p_dataset_adj")):
        vals = out[col].to_numpy(dtype=float)
        ok = np.where(np.isfinite(vals))[0]
        res = np.full(vals.size, np.nan)
        if ok.size and p_adjust != "none":
            _, padj, _, _ = multipletests(vals[ok], method=p_adjust)
            res[ok] = padj
        elif ok.size:
            res[ok] = vals[ok]
        out[adj] = res

    for _, r in out.iterrows():
        print(f"  {r['method']:<20s} offset_z={r['offset_z']:+.4f} "
              f"(native {r['offset_native']:+.4f})  p={r['p']:.3g} "
              f"[df={r['df']}]  p_adj={r['p_adj']:.3g}   "
              f"P(win)={r['p_win']:.3f} [{r['p_win_ci_low']:.3f}, {r['p_win_ci_high']:.3f}] "
              f"p={r['p_win_p']:.3g}\n"
              f"  {'':<20s} other frames: dataset-random p={r['p_dataset']:.3g} "
              f"[df={r['df_dataset']}] | dataset-fixed p={r['p_dsfixed']:.3g} "
              f"[df={r['df_dsfixed']}] | edge-naive p={r['p_edge']:.3g} [df={r['df_edge']}]\n"
              f"  {'':<20s} signed-rank over transitions p={r['wilcox_p_edge']:.3g} "
              f"(n={int(r['wilcox_n_edge'])}, rank-biserial={r['wilcox_rbc_edge']:+.3f})")
    return out


def fit_logistic(long_df, method_order, plot_cfg):
    """Paired and unpaired logistic separation of each method from the reference."""
    ref = plot_cfg.get("reference_method")
    if not ref or ref not in method_order:
        return pd.DataFrame()
    transform = str(plot_cfg.get("transform", "atanh")).lower()
    eps = _cfg_num(plot_cfg, "transform_eps", 1e-6)
    p_adjust = str(plot_cfg.get("p_adjust", "fdr_bh")).lower()
    from statsmodels.stats.multitest import multipletests

    print("\n=== Logistic separation from the reference ===")
    rows = []
    for m in method_order:
        if m == ref:
            continue
        sub = long_df[long_df["method"].isin([ref, m])]
        if sub.empty:
            continue
        # paired: complete cases, conditional on the cell
        w = sub.pivot_table(index="obs_id", columns="method", values="cbdir", aggfunc="first")
        edge_of = sub.drop_duplicates("obs_id").set_index("obs_id")["edge"]
        rec = dict(method=m, reference_method=ref)
        if ref in w.columns and m in w.columns:
            pair = w[[ref, m]].dropna()
            zm, _ = _forward_transform(pair[m].to_numpy(), transform, eps)
            zr, _ = _forward_transform(pair[ref].to_numpy(), transform, eps)
            cl_edge = edge_of.reindex(pair.index).to_numpy()
            ds_of = sub.drop_duplicates("obs_id").set_index("obs_id")["dataset"]
            cl_ds = ds_of.reindex(pair.index).to_numpy()
            pr = conditional_logit_paired(
                zm - zr, {"edge": np.char.add(
                              np.char.add(np.asarray(cl_ds, dtype=str), "___"),
                              np.asarray(cl_edge, dtype=str)),
                          "dataset": np.asarray(cl_ds, dtype=object)})
            rec.update({f"paired_{k}": v for k, v in pr.items()})
        gm, gr = sub[sub["method"] == m], sub[sub["method"] == ref]
        zm, _ = _forward_transform(gm["cbdir"].to_numpy(), transform, eps)
        zr, _ = _forward_transform(gr["cbdir"].to_numpy(), transform, eps)
        up = logit_unpaired(zm, zr, gm["dataset"], gr["dataset"],
                            gm["dataset"] + "|" + gm["edge"],
                            gr["dataset"] + "|" + gr["edge"])
        rec.update({f"unpaired_{k}": v for k, v in up.items()})
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    adj_pairs = [(f"{k}_p_{c}", f"{k}_p_adj_{c}")
                 for k in ("paired", "unpaired") for c in ("edge", "dataset")]
    for col, adj in adj_pairs:
        if col not in out.columns:
            continue
        v = out[col].to_numpy(dtype=float)
        ok = np.where(np.isfinite(v))[0]
        res = np.full(v.size, np.nan)
        if ok.size:
            res[ok] = (multipletests(v[ok], method=p_adjust)[1]
                       if p_adjust != "none" else v[ok])
        out[adj] = res
    for _, r in out.iterrows():
        print(f"  {r['method']:<20s} paired OR/0.1z={r.get('paired_or_per_0p1z', np.nan):.3f}  "
              f"edge-clust {fmt_p(r.get('paired_p_adj_edge', np.nan))} "
              f"(G={int(r.get('paired_n_clusters_edge', 0))})  |  "
              f"dataset-clust {fmt_p(r.get('paired_p_adj_dataset', np.nan))} "
              f"(G={int(r.get('paired_n_clusters_dataset', 0))})")
        print(f"  {'':<20s} unpaired OR/0.1z={r.get('unpaired_or_per_0p1z', np.nan):.3f}  "
              f"edge-clust {fmt_p(r.get('unpaired_p_adj_edge', np.nan))} "
              f"(G={int(r.get('unpaired_n_clusters_edge', 0) or 0)})  |  "
              f"dataset-clust {fmt_p(r.get('unpaired_p_adj_dataset', np.nan))} "
              f"(G={int(r.get('unpaired_n_clusters_dataset', 0) or 0)})")
    return out


_LOGISTIC_VARIANT_SHORTHAND = {
    "paired": ("", ["paired|edge", "paired|dataset"]),
    "unpaired": ("_unpaired", ["unpaired|edge", "unpaired|dataset"]),
}


def _logistic_variants(plot_cfg):
    """Resolve which logistic figures to emit, as a list of (name, panels).

    ``logistic_panels`` (a bare list of "kind|clustering" strings) still works and
    defines a single unnamed figure, so older configs keep their behaviour. When it
    is absent, ``logistic_panel_variants`` decides; each entry is either a shorthand
    ("paired", "unpaired") or a mapping with ``name`` and ``panels``.
    """
    if plot_cfg.get("logistic_panels") is not None:
        spec = plot_cfg["logistic_panels"]
        return [("", [spec] if isinstance(spec, str) else list(spec))]
    spec = plot_cfg.get("logistic_panel_variants", ["paired", "unpaired"])
    if isinstance(spec, str):
        spec = [spec]
    out, seen = [], set()
    for item in spec:
        if isinstance(item, dict):
            panels = item.get("panels")
            if panels is None:
                continue
            name = str(item.get("name", "") or "")
            panels = [panels] if isinstance(panels, str) else list(panels)
        else:
            key = str(item).strip().lower()
            if key not in _LOGISTIC_VARIANT_SHORTHAND:
                print(f"  WARNING: unknown logistic variant '{item}' - skipped")
                continue
            name, panels = _LOGISTIC_VARIANT_SHORTHAND[key]
            panels = list(panels)
        if name in seen:
            continue
        seen.add(name)
        out.append((name, panels))
    return out


def plot_logistic(logistic, method_order, plot_cfg, suffix):
    """Emit one logistic forest per requested variant (paired, unpaired, ...)."""
    if logistic.empty:
        return
    for name, panels in _logistic_variants(plot_cfg):
        _plot_logistic_one(logistic, method_order, plot_cfg, f"{name}{suffix}", panels)


def _plot_logistic_one(logistic, method_order, plot_cfg, suffix, spec):
    """Odds ratios with 95% CIs from the paired and unpaired logistic tests.

    x is the odds that the higher of two CBDir scores belongs to the method, per
    0.1 of z — 1.0 means indistinguishable from the reference. Log x-axis, so
    "twice as likely" and "half as likely" sit symmetrically about 1.
    """
    if logistic.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    ref = logistic["reference_method"].iloc[0] if "reference_method" in logistic else "reference"

    if isinstance(spec, str):
        spec = [spec]
    panels = []
    for item in spec:
        kind, _, clust = str(item).partition("|")
        kind, clust = kind.strip().lower(), (clust.strip().lower() or "edge")
        if f"{kind}_or_per_0p1z" in logistic.columns:
            panels.append((kind, clust))
    if not panels:
        return
    present = [m for m in method_order if m in set(logistic["method"])][::-1]
    figsize = tuple(plot_cfg.get("logistic_figsize")
                    or [5.6 * len(panels), 0.52 * max(len(present), 3) + 2.2])
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, dpi=dpi,
                             squeeze=False, sharey=True)

    lk = logistic.set_index("method")
    titles = {("paired", "edge"): "Paired — clustered by transition",
              ("paired", "dataset"): "Paired — clustered by dataset\n(transitions nested in datasets)",
              ("unpaired", "edge"): "Unpaired — clustered by transition",
              ("unpaired", "dataset"): "Unpaired — clustered by dataset"}
    for ax, (kind, clust) in zip(axes[0], panels):
        ncl = 0
        for y, m in enumerate(present):
            if m not in lk.index:
                continue
            r = lk.loc[m]
            est = r.get(f"{kind}_or_per_0p1z", np.nan)
            lo = r.get(f"{kind}_or_per_0p1z_low_{clust}", np.nan)
            hi = r.get(f"{kind}_or_per_0p1z_high_{clust}", np.nan)
            ncl = max(ncl, int(r.get(f"{kind}_n_clusters_{clust}", 0) or 0))
            if not np.isfinite(est):
                ax.text(1.0, y, "  not estimable", va="center", fontsize=8, color="#999999")
                continue
            col = palette.get(m, "#444444")
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2)
            ax.plot([est], [y], "o", color=col, ms=7, zorder=3,
                    markeredgecolor="#333333", markeredgewidth=0.5)
            praw = r.get(f"{kind}_p_{clust}", np.nan)
            padj = r.get(f"{kind}_p_adj_{clust}", np.nan)
            sig_adj = np.isfinite(padj) and padj < 0.05
            sig_raw = np.isfinite(praw) and praw < 0.05
            if sig_adj:
                col_t, weight = "#c2410c", "bold"
            elif sig_raw:
                col_t, weight = "#b45309", "normal"
            else:
                col_t, weight = "#777777", "normal"
            fields = plot_cfg.get("logistic_annotation", ["p", "p_adj"])
            if isinstance(fields, str):
                fields = [fields]
            fields = [str(f).strip().lower() for f in fields]
            bits = []
            if "p" in fields:
                bits.append(fmt_p(praw))
            if "p_adj" in fields:
                bits.append(fmt_p(padj, prefix="adj=") if "p" in fields else fmt_p(padj))
            ax.text(hi if np.isfinite(hi) else est, y, "  " + "  ".join(bits), va="center",
                    fontsize=7.5, fontweight=weight, color=col_t)
        ax.axvline(1.0, color="#555555", ls="--", lw=1.0, zorder=0)
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1,
                                              numticks=12))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _pos: f"{v:.2f}" if v < 10 else f"{v:.0f}"))
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present)
        ax.set_xlabel("Odds the higher score is the method's, per 0.1 z", fontsize=10)
        ax.set_title(titles.get((kind, clust), f"{kind} — {clust}") + f"   (G={ncl})",
                     fontsize=11, fontweight="bold")
    kinds = {k for k, _ in panels}
    if kinds == {"paired"}:
        scope = "\nPaired: only cells scored by both the method and the reference"
    elif kinds == {"unpaired"}:
        scope = "\nUnpaired: every cell each method scored, dataset held fixed"
    else:
        scope = ""
    fig.suptitle(plot_cfg.get("logistic_title",
                 f"Can {VALUE_LABEL} tell the method apart from {ref}?")
                 + scope
                 + "\n1.0 = indistinguishable; >1 favours the method"
                 + "   |   bold = significant after correction, "
                   "amber = nominal only, grey = neither",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90 if scope else 0.93])
    _save_figure(f"{save_base}_logistic{suffix}", dpi)


def plot_cells_vs_cbdir(long_df, method_order, dataset_order, plot_cfg, suffix):
    """Diagnostic: is a transition's CBDir related to how many cells it scored?"""
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    figsize = tuple(plot_cfg.get("diag_figsize", [11, 7]))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    g = (long_df.groupby(["dataset", "edge", "method"])
                .agg(mean_cbdir=("cbdir", "mean"), n_cells=("cbdir", "size"))
                .reset_index())
    if g.empty:
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for di, d_name in enumerate([d for d in dataset_order if d in set(g["dataset"])]):
        mk = markers[di % len(markers)]
        for m in method_order:
            sub = g[(g["dataset"] == d_name) & (g["method"] == m)]
            if sub.empty:
                continue
            ax.scatter(sub["n_cells"], sub["mean_cbdir"], marker=mk, s=46,
                       facecolor=palette.get(m, "#888888"), edgecolor="#333333",
                       linewidth=0.4, alpha=0.85, zorder=3)

    if plot_cfg.get("diag_logx", True):
        ax.set_xscale("log")
    draw_zero_line(ax, plot_cfg)

    # trend over all points, to show whether small transitions sit differently
    x = np.log10(g["n_cells"].to_numpy(dtype=float))
    y = g["mean_cbdir"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 3 and np.unique(x[ok]).size > 1:
        sl, ic, r, pv, _ = stats.linregress(x[ok], y[ok])
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(10 ** xs, ic + sl * xs, color="#c2410c", lw=1.6, ls="-", zorder=2)
        ax.text(0.02, 0.02, f"slope per 10x cells = {sl:+.3f}   r = {r:+.2f}   {fmt_p(pv)}",
                transform=ax.transAxes, fontsize=9, color="#c2410c", va="bottom")
    elif ok.sum() > 3:
        ax.text(0.02, 0.02, "every transition has the same cell count; no trend to fit",
                transform=ax.transAxes, fontsize=9, color="#666666", va="bottom")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h1 = [Patch(facecolor=palette.get(m, "#888888"), edgecolor="#333333", label=m)
          for m in method_order]
    h2 = [Line2D([], [], marker=markers[i % len(markers)], color="none",
                 markerfacecolor="#bbbbbb", markeredgecolor="#333333", markersize=8, label=d)
          for i, d in enumerate([d for d in dataset_order if d in set(g["dataset"])])]
    leg1 = ax.legend(handles=h1, title="Method", bbox_to_anchor=(1.02, 1),
                     loc="upper left", frameon=True, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=h2, title="Dataset", bbox_to_anchor=(1.02, 0.0),
              loc="lower left", frameon=True, fontsize=8)

    ax.set_xlabel(f"Cells scored in the {UNIT_LABEL}"
                  + (" (log scale)" if plot_cfg.get("diag_logx", True) else ""),
                  fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Mean {VALUE_LABEL} in the {UNIT_LABEL}", fontsize=13, fontweight="bold")
    ax.set_title(plot_cfg.get("diag_title",
                 f"Does a {UNIT_LABEL}'s {VALUE_LABEL} depend on how many cells it scored?"),
                 fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()
    _save_figure(f"{save_base}_cells_vs_cbdir{suffix}", dpi)


def plot_pooled(long_df, pooled, method_order, plot_cfg, suffix):
    """x = method, one point per (dataset, edge): its mean CBDir for that method."""
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    ylim = plot_cfg.get("ylim", [-1.0, 1.0])
    figsize = tuple(plot_cfg.get("pooled_figsize", [max(8, 1.4 * len(method_order)), 7]))
    ref = plot_cfg.get("reference_method")

    em = (long_df.groupby(["dataset", "edge", "method"], as_index=False)["cbdir"]
                 .mean().rename(columns={"cbdir": "edge_mean"}))
    present = [m for m in method_order if m in set(em["method"])]

    variants = _resolve_variants(plot_cfg.get("pooled_order_variants"),
                                 ["specified", "ranked"])
    med = em.groupby("method")["edge_mean"].median()
    orders = {
        "specified": present,
        # Rank by the median of the per-transition means, i.e. the median of the
        # points actually drawn — so the boxes read left-to-right best-to-worst.
        "ranked": sorted(present, key=lambda m: med.get(m, -np.inf), reverse=True),
    }
    if "ranked" in variants:
        print(f"  [ranked] method order by median {VALUE_LABEL} across {UNIT_LABEL_PLURAL}: "
              f"{orders['ranked']}")

    annots = plot_cfg.get("pooled_annotation_variants", ["default", "wilcoxon"])
    if isinstance(annots, str):
        annots = [annots]
    annots = [str(a).strip().lower() for a in annots] or ["default"]

    # The pooled tests already take the DATASET as the headline unit (pool_unit),
    # but the figure draws one point per transition, so a dataset cut into six
    # transitions looks six times as certain as one cut into two. The dataset
    # variant collapses each dataset first, putting the picture on the same unit
    # as the test it annotates.
    units = _resolve_point_units(plot_cfg.get("pooled_point_variants"))
    how = str(plot_cfg.get("pooled_dataset_aggregator", "median") or "median").lower()
    frames = {"transition": em}
    if "dataset" in units:
        frames["dataset"] = (em.groupby(["dataset", "method"], as_index=False)["edge_mean"]
                               .agg("median" if how.startswith("med") else "mean"))

    for unit in units:
        if unit not in frames:
            continue
        data = frames[unit]
        med_u = data.groupby("method")["edge_mean"].median()
        orders_u = {"specified": present,
                    "ranked": sorted(present, key=lambda m: med_u.get(m, -np.inf),
                                     reverse=True)}
        for variant in variants:
            for annot in annots:
                cfg = dict(plot_cfg)
                if annot == "wilcoxon":
                    cfg["pooled_annotation"] = ["wilcox"]
                    cfg["pooled_secondary_frame"] = None
                _plot_pooled_one(data, pooled, orders_u[variant], palette, cfg, ref,
                                 ylim, figsize, dpi,
                                 f"{save_base}_pooled{_variant_suffix(variant)}"
                                 f"{'_wilcoxon' if annot == 'wilcoxon' else ''}"
                                 f"{'_by_dataset' if unit == 'dataset' else ''}{suffix}",
                                 ranked=(variant == "ranked"),
                                 annot_tag=annot, point_unit=unit)


def _plot_pooled_one(em, pooled, order, palette, plot_cfg, ref, ylim, figsize, dpi,
                     out_base, ranked=False, annot_tag="default",
                     point_unit="transition"):
    """Draw one pooled figure with a given method order."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    lo_ax, hi_ax, annot_base = resolve_ylim(
        em["edge_mean"], plot_cfg, annot_frac=plot_cfg.get("pooled_headroom", 0.30))
    shape_map = (_dataset_marker_map(list(pd.unique(em["dataset"])), plot_cfg)
                 if point_unit == "dataset"
                 and plot_cfg.get("pooled_dataset_markers", True) else {})
    positions = {}
    for i, m in enumerate(order):
        vals = em.loc[em["method"] == m, "edge_mean"].to_numpy()
        positions[m] = i
        if vals.size == 0:
            continue
        bp = ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False, manage_ticks=False,
                        boxprops=dict(linewidth=plot_cfg.get("box_linewidth", 0.7),
                                      edgecolor="#333333"),
                        whiskerprops=dict(linewidth=0.7, color="#333333"),
                        capprops=dict(linewidth=0.7, color="#333333"),
                        medianprops=dict(linewidth=1.4, color="#111111"))
        for patch in bp["boxes"]:
            patch.set_facecolor(palette.get(m, "#888888"))
            patch.set_alpha(0.55)
        if plot_cfg.get("pooled_show_points", True):
            rng_p = np.random.default_rng(0)
            if point_unit == "dataset" and shape_map:
                # One point per dataset: few enough that the shape is readable,
                # and it lets a reader follow one dataset across every method.
                sub_i = em[em["method"] == m]
                for d_name, g in sub_i.groupby("dataset"):
                    ax.plot(i + (rng_p.random(len(g)) - 0.5) * 0.22, g["edge_mean"],
                            marker=shape_map.get(d_name, "o"), ms=7, linestyle="none",
                            color=palette.get(m, "#888888"), markeredgecolor="#333333",
                            markeredgewidth=0.45, alpha=0.95, zorder=3)
            else:
                jit = (rng_p.random(vals.size) - 0.5) * 0.28
                ax.plot(i + jit, vals, "o", ms=3.5, color=palette.get(m, "#888888"),
                        markeredgecolor="#333333", markeredgewidth=0.3, alpha=0.9,
                        zorder=3)

    draw_zero_line(ax, plot_cfg)

    # annotate the pooled test above each non-reference box
    if not pooled.empty:
        pk = pooled.set_index("method")
        fields = plot_cfg.get("pooled_annotation", ["p_adj", "pwin"])
        if isinstance(fields, str):
            fields = [fields]
        fields = [str(f).strip().lower() for f in fields]

        headline = pooled["pool_unit"].iloc[0]
        secondary = plot_cfg.get("pooled_secondary_frame", "auto")
        if isinstance(secondary, str) and secondary.lower() == "auto":
            secondary = "dataset_fixed" if headline != "dataset_fixed" else "dataset"
        elif secondary in (None, False, "none", "null"):
            secondary = None

        top = annot_base
        for m in order:
            if m == ref or m not in pk.index:
                continue
            r = pk.loc[m]
            lines = []
            if "stars" in fields:
                lines.append(pvalue_to_stars(r.get("p_adj", np.nan)))
            if "p" in fields:
                lab_h = _FRAME_COLS.get(headline, (None, None, headline))[2]
                lines.append(f"{lab_h} {fmt_p(r.get('p', np.nan))} (raw)")
            if "p_adj" in fields:
                lab_h = _FRAME_COLS.get(headline, (None, None, headline))[2]
                lines.append(f"{lab_h} {fmt_p(r.get('p_adj', np.nan))}")
            if "offset" in fields:
                lines.append(f"\u0394={r.get('offset_native', np.nan):+.3f}")
            if "pwin" in fields:
                lines.append(f"P(win)={r.get('p_win', np.nan):.2f}")
            if "n" in fields:
                lines.append(f"n={int(r.get('n_edges', 0))} edges")
            if "wilcox" in fields:
                lines.append(f"signed-rank\n{fmt_p(r.get('wilcox_p_edge_adj', np.nan))}")
                lines.append(f"r={r.get('wilcox_rbc_edge', np.nan):+.2f}")
            if secondary and secondary in _FRAME_COLS:
                pcol, dcol, lab = _FRAME_COLS[secondary]
                lines.append(f"{lab} {fmt_p(r.get(pcol, np.nan))}")
            sig_col = "wilcox_p_edge_adj" if "wilcox" in fields else "p_adj"
            sig = np.isfinite(r.get(sig_col, np.nan)) and r[sig_col] < 0.05
            ax.text(positions[m], top, "\n".join(lines), ha="center", va="bottom",
                    fontsize=plot_cfg.get("pooled_annotation_fontsize", 7.5),
                    fontweight="bold" if sig else "normal",
                    color="#c2410c" if sig else "#555555", linespacing=1.35)
        if ref in positions:
            ax.text(positions[ref], top, "reference", ha="center", va="bottom",
                    fontsize=plot_cfg.get("pooled_annotation_fontsize", 7.5),
                    style="italic", color="#666666")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(lo_ax, hi_ax)
    unit = pooled["pool_unit"].iloc[0] if not pooled.empty else "dataset"
    title = plot_cfg.get("pooled_title") or (
        f"{VALUE_LABEL} by dataset, pooled across datasets" if point_unit == "dataset"
        else f"{VALUE_LABEL} per {UNIT_LABEL}, pooled across datasets")
    if ranked:
        title += " (methods ranked by median)"
    if annot_tag == "wilcoxon":
        title += " — paired signed-rank over transitions"
    point_txt = (f"each point = one dataset ({UNIT_LABEL_PLURAL} collapsed first)"
                 if point_unit == "dataset" else f"each point = one {UNIT_LABEL}")
    ax.set_title(title
                 + f"\n({point_txt}; "
                 + {"dataset": "dataset as a random effect",
                    "dataset_fixed": "dataset fixed, edge random",
                    "edge": "edges pooled, dataset ignored"}.get(unit, unit)
                 + f", vs {ref})",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Method", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Mean {VALUE_LABEL} per "
                  + ("dataset" if point_unit == "dataset" else UNIT_LABEL),
                  fontsize=13, fontweight="bold")
    rot = plot_cfg.get("xlabel_rotation", 45)
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot != 0 else "center")
    n_leg = 0
    if shape_map:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker=mk, color="#555555", linestyle="none",
                          markersize=7, markeredgecolor="#333333", markeredgewidth=0.45,
                          label=str(dn)) for dn, mk in shape_map.items()]
        ncol = _cfg_int(plot_cfg, "pooled_dataset_legend_ncol", 0) or min(len(handles), 4)
        n_leg = int(np.ceil(len(handles) / ncol))
        fig.legend(handles=handles, title="Dataset", fontsize=8, title_fontsize=9,
                   loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False,
                   ncol=ncol)
    fig.tight_layout(rect=[0, 0.02 + 0.05 * n_leg, 1, 1] if n_leg else None)
    _save_figure(out_base, dpi)


def plot_forest(results, method_order, dataset_order, plot_cfg, suffix):
    """Offsets with 95% CIs, one panel per dataset."""
    if results.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    scale = str(plot_cfg.get("forest_scale", "native")).lower()
    col, lo_col, hi_col = (("offset_native", "ci_native_low", "ci_native_high")
                           if scale == "native" else ("offset_z", "ci_z_low", "ci_z_high"))

    datasets = [d for d in dataset_order if d in set(results["dataset"])]
    if not datasets:
        return
    ncol = min(3, len(datasets))
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 0.55 * max(len(method_order), 3) * nrow + 1.6),
                             dpi=dpi, squeeze=False)
    sns.set_theme(style="whitegrid")

    for ax, d_name in zip(axes.flat, datasets):
        sub = results[results["dataset"] == d_name]
        plot_methods = [m for m in method_order if m in set(sub["method"])][::-1]
        for y, m in enumerate(plot_methods):
            r = sub[sub["method"] == m].iloc[0]
            if not np.isfinite(r.get(col, np.nan)):
                continue
            lo, hi = r.get(lo_col, np.nan), r.get(hi_col, np.nan)
            ax.plot([lo, hi], [y, y], color=palette.get(m, "#444444"), lw=2.0, solid_capstyle="round")
            ax.plot([r[col]], [y], "o", color=palette.get(m, "#444444"), ms=7, zorder=3)
            stars = pvalue_to_stars(r.get("p_adj", np.nan))
            sig = stars != "ns"
            if plot_cfg.get("forest_show_p", True):
                ax.text(hi, y, f"  {fmt_p(r.get('p_adj', np.nan))}", va="center",
                        fontsize=8, fontweight="bold" if sig else "normal",
                        color="#c2410c" if sig else "#666666")
            elif sig:
                ax.text(hi, y, f"  {stars}", va="center", fontsize=9,
                        fontweight="bold", color="#d95f02")
        ax.axvline(0.0, color="#555555", linestyle="--", lw=0.9)
        ax.set_yticks(range(len(plot_methods)))
        ax.set_yticklabels(plot_methods)
        ax.set_title(d_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{VALUE_LABEL} offset vs {results['reference_method'].iloc[0]}"
                      + ("" if scale == "native" else " (z units)"), fontsize=10)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)

    fig.suptitle(plot_cfg.get("forest_title", "Per-method offset vs reference (95% CI)"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(f"{save_base}_lmm_forest{suffix}", dpi)


def plot_diagnostics(residuals, method_order, plot_cfg, suffix):
    """Residual QQ and residual-vs-fitted, per dataset."""
    if residuals.empty:
        return
    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    palette = build_palette(method_order, plot_cfg.get("method_colors"))
    dpi = plot_cfg.get("dpi", 300)
    datasets = list(pd.unique(residuals["dataset"]))

    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4.0 * len(datasets)),
                             dpi=dpi, squeeze=False)
    sns.set_theme(style="whitegrid")
    for i, d_name in enumerate(datasets):
        sub = residuals[residuals["dataset"] == d_name]
        ax = axes[i][0]
        for m in [x for x in method_order if x in set(sub["method"])]:
            r = np.sort(sub.loc[sub["method"] == m, "resid"].to_numpy())
            if r.size < 3:
                continue
            q = stats.norm.ppf((np.arange(1, r.size + 1) - 0.5) / r.size)
            ax.plot(q, (r - r.mean()) / (r.std() or 1.0), ".", ms=2,
                    color=palette.get(m, "#444444"), label=m, alpha=0.5)
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", lw=0.9)
        ax.set_title(f"{d_name} — residual QQ", fontsize=11, fontweight="bold")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Standardised residuals")
        if i == 0:
            ax.legend(fontsize=7, markerscale=3)

        ax = axes[i][1]
        for m in [x for x in method_order if x in set(sub["method"])]:
            s = sub[sub["method"] == m]
            ax.plot(s["fitted"], s["resid"], ".", ms=2,
                    color=palette.get(m, "#444444"), alpha=0.5)
        ax.axhline(0, color="#555555", ls="--", lw=0.9)
        ax.set_title(f"{d_name} — residual vs fitted", fontsize=11, fontweight="bold")
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")

    fig.tight_layout()
    _save_figure(f"{save_base}_lmm_diagnostics{suffix}", dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_cbdir(global_config_path, plot_config_path, str_suffix=None, log_file=None):
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML is required. Install it with `pip install pyyaml`.") from e

    plot_cfg = {}
    if plot_config_path and os.path.exists(plot_config_path):
        with open(plot_config_path, "r") as fh:
            plot_cfg = yaml.safe_load(fh) or {}

    if log_file is None:
        log_file = plot_cfg.get("log_file", None)
    log_path = _resolve_log_file(log_file)

    log_fh = None
    if log_path:
        log_dir = os.path.dirname(os.path.abspath(log_path))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_fh = _FilteredFile(open(log_path, "a"))
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)

    try:
        _run(global_config_path, plot_cfg, str_suffix, log_path)
    finally:
        if log_fh is not None:
            log_fh.flush()
            log_fh.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"Log written to: {os.path.abspath(log_path)}")


def _run(global_config_path, plot_cfg, str_suffix, log_path=None):
    """Run the figure suite on CBDir, then replay it on the alignment.

    The alignment is CBDir divided by the cone's mean resultant length, a
    strictly positive per-cell quantity, so the second pass is a rescaling and
    not a second set of claims. Worth being precise about what that implies:
    every SIGN-based result is identical between the two folders — transitions
    above zero, the reversal table, sign tests — because the same Rbar_i divides
    both sides of every paired difference. Only magnitude- and rank-based
    statistics move, and they move because cone tightness no longer inflates the
    tight-cone transitions.

    The replay reuses the same functions rather than copying them, so the two
    folders cannot drift apart. Two analyses are suppressed in it: the ceiling
    (the ceiling of an alignment is 1 by construction) and the coherence control
    (it belongs with the raw metric it was built to interrogate).
    """
    _run_body(global_config_path, plot_cfg, str_suffix, log_path)

    if not plot_cfg.get("align_plots", True) or plot_cfg.get("_metric") == "alignment":
        return
    base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    sub = str(plot_cfg.get("align_subdir", "cbdir_align_plots") or "")
    acfg = dict(plot_cfg)
    acfg["_metric"] = "alignment"
    acfg["save_path"] = os.path.join(os.path.dirname(os.path.abspath(base)), sub,
                                     os.path.basename(base)) if sub else base
    for k in ("ceiling_test", "ceiling_plot", "iccoh_test", "iccoh_plot",
              "iccoh_stability_plot", "gene_bias_test", "gene_bias_plot"):
        acfg[k] = False
    print(f"\n\n=== Replaying the suite on the alignment (CBDir / Rbar) "
          f"-> {os.path.dirname(acfg['save_path'])} ===")
    try:
        with vocabulary(value_label=str(plot_cfg.get("align_value_label", "Alignment"))):
            _run_body(global_config_path, acfg, str_suffix, log_path)
    except Exception as e:
        print(f"  WARNING: the alignment replay failed: {type(e).__name__}: {e}")


def _run_body(global_config_path, plot_cfg, str_suffix, log_path=None, preloaded=None):
    """One pass of the run, with stdout/stderr already tee'd when logging is on.

    `preloaded` lets another pipeline drive this suite on a table it shaped
    itself, as (long_df, methods, dataset_order, edges_by_dataset); the loader
    and the alignment pass are then skipped and `global_config_path` is unused.
    plot_velocity_confidence.py uses it to run these figures on velocity
    confidence with `cluster` standing in for `edge`.

    This hook exists so that no caller has to re-copy the sequence below. A copy
    silently stops gaining every stage added here afterwards, which is exactly
    what happened before it existed: the confidence pipeline carried a duplicate
    that had fallen several figures behind. Add new stages HERE, once.

    Stages needing columns the caller may not have (resultant_length for the
    ceiling, iccoh for the coherence control) skip themselves, so a partial table
    is fine.
    """
    import yaml

    print(f"=== CBDir plotting run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    if log_path:
        print(f"Log file: {os.path.abspath(log_path)}")
    dataset_dir_paths = {}
    if preloaded is not None:
        long_df, methods, dataset_order, edges_by_dataset = preloaded
        long_df = long_df.copy()
        print(f"=== Using a caller-supplied long table "
              f"({len(long_df)} rows, {len(methods)} methods, "
              f"{len(dataset_order)} datasets) ===")
    else:
        print("=== Loading CBDir long tables ===")
        long_df, methods, dataset_order, dataset_dir_paths, edges_by_dataset = load_cbdir_long(
            global_config_path, str_suffix=str_suffix)

    if preloaded is None and plot_cfg.get("_metric") == "alignment":
        if "alignment" not in long_df.columns or long_df["alignment"].isna().all():
            print("  Skipping the alignment pass: the long tables carry no alignment "
                  "column (re-run compute_cbdir_run.py to produce it)")
            return
        n0 = len(long_df)
        long_df = long_df[long_df["alignment"].notna()].copy()
        long_df["cbdir"] = long_df["alignment"].to_numpy()
        print(f"  Plotting the ALIGNMENT (CBDir / Rbar) in place of CBDir "
              f"({len(long_df)}/{n0} rows carry one)")

    if str_suffix is None and global_config_path is not None:
        with open(global_config_path, "r") as fh:
            str_suffix = (yaml.safe_load(fh) or {}).get("str_suffix", None)
    suffix = _suffix_str(str_suffix)

    method_order = compute_grouped_method_order(
        long_df, plot_cfg.get("method_groups"), plot_cfg.get("method_order"), methods)
    long_df = long_df[long_df["method"].isin(method_order)].copy()
    print(f"  Methods : {method_order}")
    print(f"  Datasets: {dataset_order}")

    results = residuals = per_edge = pooled = pd.DataFrame()
    per_edge_all = []
    pwin_lookup = None
    if plot_cfg.get("run_lmm", True):
        results, residuals, per_edge = fit_lmm(long_df, method_order, dataset_order, plot_cfg)
        if not results.empty and "p_win" in results.columns:
            pwin_lookup = {(r["dataset"], r["method"]): r["p_win"]
                           for _, r in results.iterrows() if np.isfinite(r.get("p_win", np.nan))}
        pairing_cfg = str(plot_cfg.get("pairing", "both")).lower()
        want = {"paired": ["paired"], "unpaired": ["unpaired"],
                "both": ["paired", "unpaired"]}.get(pairing_cfg, ["paired", "unpaired"])
        tables, pooled_parts = {}, []
        if "paired" in want and not per_edge.empty:
            tables["paired"] = per_edge
        if "unpaired" in want:
            up = build_per_edge_unpaired(
                long_df, plot_cfg.get("reference_method"),
                str(plot_cfg.get("transform", "atanh")).lower(),
                _cfg_num(plot_cfg, "transform_eps", 1e-6))
            if not up.empty:
                tables["unpaired"] = up
                kept = (per_edge.merge(up, on=["dataset", "method", "edge"],
                                       suffixes=("_p", "_u"))
                        if not per_edge.empty else pd.DataFrame())
                if not kept.empty:
                    rho = kept["mean_diff_z_p"].corr(kept["mean_diff_z_u"])
                    print(f"\n  Paired vs unpaired per-transition offsets: "
                          f"r = {rho:.3f} over {len(kept)} transitions "
                          f"(high r = the complete-case subsetting does not matter)")
        for lab, tbl in tables.items():
            cfg2 = dict(plot_cfg); cfg2["_pairing_label"] = lab
            part = fit_pooled(tbl, cfg2)
            if not part.empty:
                pooled_parts.append(part)
            tbl2 = tbl.copy(); tbl2["pairing"] = lab
            per_edge_all.append(tbl2)
        if pooled_parts:
            pooled = pd.concat(pooled_parts, ignore_index=True)

    print("\n=== Aggregated plots ===")
    plot_aggregated(long_df, method_order, dataset_order, plot_cfg, suffix, pwin=pwin_lookup)

    logistic = pd.DataFrame()
    if plot_cfg.get("logistic_test", True):
        try:
            logistic = fit_logistic(long_df, method_order, plot_cfg)
        except Exception as e:
            print(f"  WARNING: logistic tests failed: {type(e).__name__}: {e}")

    if not logistic.empty and plot_cfg.get("logistic_plot", True):
        print("\n=== Logistic figure ===")
        plot_logistic(logistic, method_order, plot_cfg, suffix)

    if plot_cfg.get("cells_vs_cbdir_plot", True):
        print("\n=== Diagnostic: cells per transition vs CBDir ===")
        plot_cells_vs_cbdir(long_df, method_order, dataset_order, plot_cfg, suffix)

    print("\n=== Per-dataset plots ===")
    plot_per_dataset(long_df, method_order, dataset_order, edges_by_dataset, plot_cfg, suffix)

    stability, levels = pd.DataFrame(), pd.DataFrame()
    if plot_cfg.get("stability_test", True):
        try:
            stability, levels = fit_stability(long_df, method_order, plot_cfg)
        except Exception as e:
            print(f"  WARNING: stability analysis failed: {type(e).__name__}: {e}")
    if not stability.empty and plot_cfg.get("stability_plot", True):
        print("\n=== Stability figure ===")
        plot_stability(stability, levels, method_order, plot_cfg, suffix,
                       dataset_order)

    if plot_cfg.get("head_to_head"):
        try:
            h2h_e, h2h_d, h2h_st = fit_head_to_head(long_df, plot_cfg)
            if h2h_st and plot_cfg.get("head_to_head_plot", True):
                plot_head_to_head(h2h_e, h2h_d, h2h_st, plot_cfg, suffix)
            if h2h_st:
                base_h = _clean_save_base(plot_cfg.get("save_path",
                                                       "./results/cbdir/cbdir_plot"))
                os.makedirs(os.path.dirname(os.path.abspath(base_h)), exist_ok=True)
                pd.DataFrame([h2h_st]).to_csv(f"{base_h}_head_to_head{suffix}.csv",
                                              index=False)
                h2h_e.to_csv(f"{base_h}_head_to_head_transitions{suffix}.csv",
                             index=False)
        except Exception as e:
            print(f"  WARNING: head-to-head failed: {type(e).__name__}: {e}")

    ceil_pts, ceil_summary = pd.DataFrame(), pd.DataFrame()
    if plot_cfg.get("ceiling_test", True):
        try:
            ceil_pts, ceil_summary = fit_cbdir_ceiling(long_df, method_order, plot_cfg)
        except Exception as e:
            print(f"  WARNING: ceiling analysis failed: {type(e).__name__}: {e}")
    if not ceil_summary.empty and plot_cfg.get("ceiling_plot", True):
        print("\n=== Achieved-vs-attainable figure ===")
        try:
            plot_cbdir_ceiling(ceil_pts, ceil_summary, method_order, plot_cfg, suffix)
        except Exception as e:
            print(f"  WARNING: ceiling figure failed: {type(e).__name__}: {e}")

    iccoh_pts, iccoh_summary = pd.DataFrame(), pd.DataFrame()
    if plot_cfg.get("iccoh_test", True):
        try:
            iccoh_pts, iccoh_summary = fit_cbdir_vs_iccoh(long_df, method_order, plot_cfg)
        except Exception as e:
            print(f"  WARNING: CBDir-vs-ICCoh failed: {type(e).__name__}: {e}")
    if not iccoh_summary.empty and plot_cfg.get("iccoh_plot", True):
        print("\n=== CBDir vs ICCoh figure ===")
        try:
            plot_cbdir_vs_iccoh(iccoh_pts, iccoh_summary, method_order, plot_cfg, suffix)
        except Exception as e:
            print(f"  WARNING: CBDir-vs-ICCoh figure failed: {type(e).__name__}: {e}")

    if plot_cfg.get("iccoh_stability_plot", True):
        print("\n=== ICCoh stability figures (heatmap + mean vs SD) ===")
        try:
            plot_iccoh_stability(long_df, method_order, plot_cfg, suffix, dataset_order)
        except Exception as e:
            print(f"  WARNING: ICCoh stability figures failed: {type(e).__name__}: {e}")

    gene_bias = {}
    if plot_cfg.get("gene_bias_test", True) and preloaded is None:
        try:
            g_tab, g_clu = load_iccoh_gene_tables(dataset_dir_paths, dataset_order, str_suffix)
            gene_bias = fit_iccoh_gene_bias(long_df, method_order, plot_cfg,
                                            genes=g_tab, gene_clusters=g_clu)
        except Exception as e:
            print(f"  WARNING: ICCoh gene-count analysis failed: {type(e).__name__}: {e}")
    if gene_bias and plot_cfg.get("gene_bias_plot", True):
        print("\n=== ICCoh gene-count bias figure ===")
        try:
            plot_iccoh_gene_bias(gene_bias, method_order, plot_cfg, suffix)
        except Exception as e:
            print(f"  WARNING: ICCoh gene-count figure failed: {type(e).__name__}: {e}")

    stability_ds = pd.DataFrame()
    if not levels.empty and plot_cfg.get("stability_per_dataset_test", True):
        try:
            stability_ds = fit_stability_per_dataset(levels, method_order, plot_cfg)
        except Exception as e:
            print(f"  WARNING: per-dataset stability failed: {type(e).__name__}: {e}")
    if not stability_ds.empty and plot_cfg.get("stability_per_dataset_plot", True):
        print("\n=== Per-dataset stability figures (one dot per transition) ===")
        try:
            plot_stability_per_dataset(stability_ds, levels, method_order, plot_cfg,
                                       suffix, dataset_order)
        except Exception as e:
            print(f"  WARNING: per-dataset stability figures failed: "
                  f"{type(e).__name__}: {e}")

    leftout_edge, leftout_summary = pd.DataFrame(), pd.DataFrame()
    if plot_cfg.get("leftout_test", True):
        print("\n=== Left-out cells (is the paired subset representative?) ===")
        try:
            leftout_edge, leftout_summary = fit_leftout(long_df, method_order, plot_cfg)
        except Exception as e:
            print(f"  WARNING: left-out test failed: {type(e).__name__}: {e}")
    if not leftout_edge.empty and plot_cfg.get("leftout_plot", True):
        plot_leftout_summary(leftout_summary, method_order, plot_cfg, suffix)
        if plot_cfg.get("leftout_coverage_plot", True):
            plot_leftout_coverage(leftout_edge, method_order, plot_cfg, suffix)
        if plot_cfg.get("leftout_by_method_plot", True):
            plot_leftout_by_method(long_df, method_order, dataset_order,
                                   edges_by_dataset, leftout_edge, plot_cfg, suffix)

    if plot_cfg.get("per_method_plots", True):
        print("\n=== Per-method panels (method vs reference, every transition) ===")
        try:
            pm_tests = per_method_edge_tests(long_df, method_order, plot_cfg)
            plot_per_method(long_df, method_order, dataset_order, edges_by_dataset,
                            pm_tests, plot_cfg, suffix)
        except Exception as e:
            print(f"  WARNING: per-method panels failed: {type(e).__name__}: {e}")

    if not pooled.empty:
        print("\n=== Pooled figures ===")
        for lab, gp in pooled.groupby("pairing", sort=False):
            plot_pooled(long_df, gp, method_order, plot_cfg,
                        (f"_{lab}" if lab != "paired" else "") + suffix)

    save_base = _clean_save_base(plot_cfg.get("save_path", "./results/cbdir/cbdir_plot"))
    out_dir = os.path.dirname(os.path.abspath(save_base))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if per_edge_all:
        per_edge = pd.concat(per_edge_all, ignore_index=True)
    for df_out, tag in ((results, "lmm"), (pooled, "pooled"), (per_edge, "per_edge"),
                        (logistic, "logistic"), (leftout_edge, "leftout"),
                        (leftout_summary, "leftout_summary"),
                        (stability, "stability"),
                        (stability_ds, "stability_per_dataset"),
                        (iccoh_summary, "cbdir_vs_iccoh"),
                        (iccoh_pts, "cbdir_vs_iccoh_points"),
                        (ceil_summary, "ceiling"),
                        (ceil_pts, "ceiling_points"),
                        (gene_bias.get("tests", pd.DataFrame()), "iccoh_gene_bias"),
                        (gene_bias.get("points", pd.DataFrame()), "iccoh_gene_bias_points"),
                        (gene_bias.get("per_method", pd.DataFrame()),
                         "iccoh_gene_bias_per_method"),
                        (gene_bias.get("ranks", pd.DataFrame()), "iccoh_gene_bias_ranks"),
                        (gene_bias.get("dose", pd.DataFrame()), "iccoh_gene_dose")):
        if not df_out.empty:
            pth = f"{save_base}_{tag}{suffix}.csv"
            df_out.to_csv(pth, index=False)
            print(f"  Saved: {pth}  ({df_out.shape[0]} rows)")
    if not results.empty and plot_cfg.get("forest_plot", True):
        print("\n=== Forest plot ===")
        plot_forest(results, method_order, dataset_order, plot_cfg, suffix)
    if plot_cfg.get("diagnostics_plot", True) and not residuals.empty:
        print("\n=== Diagnostics ===")
        plot_diagnostics(residuals, method_order, plot_cfg, suffix)

    print("\n=== Done ===")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot CBDir across methods and datasets, and fit per-method "
                    "reference-offset mixed models.")
    parser.add_argument("--global-config", required=True,
                        help="Path to the CBDir global YAML config.")
    parser.add_argument("--plot-config", required=True,
                        help="Path to the plotting YAML config.")
    parser.add_argument("--str-suffix", "--suffix", dest="str_suffix", default=None,
                        help="Optional suffix on input/output filenames (e.g. '_v2').")
    parser.add_argument("--log-file", dest="log_file", default=None,
                        help="Write a run log here (overrides log_file in the plot "
                             "config). Pass 'auto' for a timestamped filename.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    plot_cbdir(args.global_config, args.plot_config, str_suffix=args.str_suffix,
               log_file=args.log_file)
    sys.exit(0)
