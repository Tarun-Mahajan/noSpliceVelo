"""Negative control for the CBDir smoke test: permute nSV velocities across cells.

Reads <dir>/adata_nosplicevelo_stream.h5ad, permutes the rows of the velocity
layer (and drops its velocity graph so compute_cbdir_run.py rebuilds it), and
writes <dir>/adata_shuffled_control.h5ad. Expected CBDir: about 0.

    python make_shuffled_control.py <stream_dir> [<stream_dir> ...]
"""
import os
import sys

import anndata as ad
import numpy as np

VKEY = "velocity_mu_vote"

for d in sys.argv[1:]:
    a = ad.read_h5ad(os.path.join(d, "adata_nosplicevelo_stream.h5ad"))
    rng = np.random.default_rng(0)
    a.layers[VKEY] = np.asarray(a.layers[VKEY])[rng.permutation(a.n_obs)]
    for k in (f"{VKEY}_graph", f"{VKEY}_graph_neg", f"{VKEY}_params"):
        a.uns.pop(k, None)
    for k in [k for k in a.obsm if k.startswith(VKEY + "_")]:
        del a.obsm[k]
    out = os.path.join(d, "adata_shuffled_control.h5ad")
    a.write_h5ad(out)
    print("wrote", out)
