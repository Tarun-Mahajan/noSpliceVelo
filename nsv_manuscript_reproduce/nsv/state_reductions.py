"""Extra state reductions for noSpliceVelo velocities (pure numpy; no torch).

The model gives, per cell and gene, a posterior over 4 kinetic states
(0 up, 1 upper steady, 2 down, 3 lower steady) and per-state moments/velocities.
Two reductions already exist:

  argmax_stable : velocity of the most probable state. On-manifold and unbiased
                  in sign where the posterior is clear, but where it is near a tie
                  neighbouring cells flip between states, so the field is noisy
                  (low velocity consistency).
  soft          : posterior-weighted mixture. Smooth, but mixes up- and down-state
                  velocities, which can flip the sign (e.g. rising genes given a
                  negative velocity when the down state carries some weight).

This module adds two in-between reductions:

  argmax_knn    : argmax of the posterior after averaging it over each cell's kNN
                  neighbourhood (row-normalised connectivities + self loop, the same
                  kind of graph used to smooth the moments). Neighbouring cells then
                  share a state unless the evidence actually changes, which removes
                  the argmax sign noise without mixing velocities across states.
                  `n_iter` repeats the averaging (more spatial regularisation).
  tempered      : mixture with weights p_s^tau / sum_s' p_s'^tau. tau = 1 is `soft`,
                  tau -> inf approaches argmax; tau in 2..8 interpolates.

All functions take the bootstrap-averaged probabilities `probA` (N, G, S) and the
per-state arrays in canonical order [up, up_f, down, down_f], each (N, G)
(constants allowed), exactly as nosplicevelo_ll_params builds them.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from scipy import sparse as sp


def row_normalised_knn(connectivities, add_self: bool = True):
    """W = row-normalise(C (+ I)). C: (N, N) sparse kNN connectivities."""
    C = sp.csr_matrix(connectivities, dtype=np.float64)
    if add_self:
        C = C + sp.identity(C.shape[0], format="csr", dtype=np.float64)
    rs = np.asarray(C.sum(1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ C


def smooth_posterior(probA: np.ndarray, W, n_iter: int = 1) -> np.ndarray:
    """Average each gene's state posterior over the kNN graph: P <- W P, n_iter times."""
    P = np.asarray(probA, dtype=np.float64)
    out = np.empty_like(P)
    for s in range(P.shape[2]):
        x = P[:, :, s]
        for _ in range(max(int(n_iter), 1)):
            x = W @ x
        out[:, :, s] = x
    return out


def _select(state: np.ndarray, arrays: Sequence) -> np.ndarray:
    out = np.zeros(state.shape, dtype=np.result_type(*[np.asarray(a).dtype for a in arrays], np.float32))
    for k, arr in enumerate(arrays):
        m = state == k
        out[m] = np.asarray(arr)[m] if np.ndim(arr) else arr
    return out


def _mix(w: np.ndarray, arrays: Sequence) -> np.ndarray:
    out = np.zeros(w.shape[:2], dtype=np.float64)
    for k, arr in enumerate(arrays):
        out += w[:, :, k] * arr
    return out


def tempered_weights(probA: np.ndarray, tau: float, eps: float = 1e-12) -> np.ndarray:
    lp = np.log(np.clip(np.asarray(probA, dtype=np.float64), eps, None)) * float(tau)
    lp -= lp.max(axis=2, keepdims=True)                     # stable softmax of tau*log p
    w = np.exp(lp)
    return w / w.sum(axis=2, keepdims=True)


def extra_reductions(probA: np.ndarray, muA: Sequence, varA: Sequence, vmuA: Sequence,
                     vvarA: Sequence, reductions: Iterable[str], W=None, n_iter: int = 1,
                     taus: Iterable[float] = (4.0,), verbose: bool = True) -> Dict[str, np.ndarray]:
    """Return {'<quantity>_<name>': (N, G)} for quantity in mu, var, velo_mu, velo_var.

    names: 'argmax_knn' and 'tempered<tau>' (e.g. 'tempered4' for tau=4; tau printed
    with %g). Also returns 'state_change_frac_argmax_knn' (scalar array): fraction of
    (cell, gene) entries whose state differs from plain argmax.
    """
    reductions = list(reductions or [])
    out: Dict[str, np.ndarray] = {}
    base_state = np.argmax(probA, axis=2)
    for red in reductions:
        if red == "argmax_knn":
            if W is None:
                raise ValueError("argmax_knn needs the kNN weight matrix W")
            Ps = smooth_posterior(probA, W, n_iter=n_iter)
            st = np.argmax(Ps, axis=2)
            name = "argmax_knn"
            out[f"mu_{name}"] = _select(st, muA)
            out[f"var_{name}"] = _select(st, varA)
            out[f"velo_mu_{name}"] = _select(st, vmuA)
            out[f"velo_var_{name}"] = _select(st, vvarA)
            frac = float(np.mean(st != base_state))
            out[f"state_change_frac_{name}"] = np.array(frac)
            if verbose:
                print(f"[state_reductions] argmax_knn (n_iter={n_iter}): {frac:.1%} of cell-gene "
                      f"entries change state vs plain argmax")
        elif red == "tempered":
            for tau in taus:
                w = tempered_weights(probA, tau)
                name = f"tempered{float(tau):g}"
                out[f"mu_{name}"] = _mix(w, muA)
                out[f"var_{name}"] = _mix(w, varA)
                out[f"velo_mu_{name}"] = _mix(w, vmuA)
                out[f"velo_var_{name}"] = _mix(w, vvarA)
                if verbose:
                    print(f"[state_reductions] tempered tau={tau:g}: mean max weight "
                          f"{float(w.max(axis=2).mean()):.3f} (soft {float(np.asarray(probA).max(axis=2).mean()):.3f})")
        else:
            raise ValueError(f"unknown extra reduction {red!r} (use 'argmax_knn' and/or 'tempered')")
    return out


def selftest(seed: int = 0) -> None:
    """Checks: tau=1 equals soft; large tau equals argmax; argmax_knn with identity W equals
    argmax; smoothing removes isolated state flips in a chain of cells."""
    rng = np.random.default_rng(seed)
    N, G, S = 60, 5, 4
    p = rng.dirichlet(np.ones(S), size=(N, G))
    arrs = [rng.normal(size=(N, G)) for _ in range(S)]
    vm = (arrs[0], np.zeros((N, G)), arrs[2], np.zeros((N, G)))
    soft = _mix(p, vm)
    r = extra_reductions(p, arrs, arrs, vm, vm, ["tempered"], taus=[1.0, 1e5], verbose=False)
    assert np.allclose(r["velo_mu_tempered1"], soft), "tau=1 must equal soft"
    hard = _select(np.argmax(p, 2), vm)
    assert np.allclose(r["velo_mu_tempered100000"], hard, atol=1e-6), "large tau must equal argmax"
    I = sp.identity(N, format="csr")
    r2 = extra_reductions(p, arrs, arrs, vm, vm, ["argmax_knn"], W=I, verbose=False)
    assert np.allclose(r2["velo_mu_argmax_knn"], hard), "identity W must equal argmax"
    # chain: cells 0..N-1, true state up (0) everywhere but posterior a near tie, every 3rd flipped
    q = np.zeros((N, 1, S))
    q[:, 0, 0], q[:, 0, 2] = 0.52, 0.48
    q[::3, 0, 0], q[::3, 0, 2] = 0.48, 0.52
    A = sp.diags([1.0, 1.0], [-1, 1], shape=(N, N), format="csr")
    W = row_normalised_knn(A)
    up = np.ones((N, 1)); dn = -np.ones((N, 1)); z = np.zeros((N, 1))
    r3 = extra_reductions(q, [up] * 4, [up] * 4, (up, z, dn, z), (up, z, dn, z), ["argmax_knn"],
                          W=W, verbose=False)
    plain = _select(np.argmax(q, 2), (up, z, dn, z))
    assert (plain < 0).sum() == N // 3, "setup: every 3rd cell flipped"
    assert (r3["velo_mu_argmax_knn"] < 0).sum() == 0, "kNN smoothing should remove isolated flips"
    print("state_reductions selftest: 4/4 checks passed")


if __name__ == "__main__":
    selftest()
