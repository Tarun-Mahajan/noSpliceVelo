import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.stats import linregress
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# --- Helper Functions (Must be outside the class for Joblib pickling) ---

def _fit_ellipse_robust(x, y):
    """Fits an ellipse and actively rejects pathological orthogonal 'cigar' fits."""
    n = len(x)
    if n < 6: return None, None
        
    # 1. Normalize data to [0, 1] to prevent eigenvalue explosion
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    x_scale, y_scale = (x_max - x_min) or 1, (y_max - y_min) or 1
    x_n, y_n = (x - x_min) / x_scale, (y - y_min) / y_scale

    # 2. Fit normalized ellipse (Fitzgibbon constraint: 4AC - B^2 = 1)
    D = np.hstack([x_n[:, None]**2, x_n[:, None]*y_n[:, None], y_n[:, None]**2, 
                   x_n[:, None], y_n[:, None], np.ones_like(x_n[:, None])])
    S = np.dot(D.T, D)
    C = np.zeros((6, 6))
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1 
    
    try:
        evals, evecs = eig(S, C)
        pos_idx = np.where((evals > 0) & (np.isfinite(evals)))[0]
        if len(pos_idx) == 0: return None, None
        idx = pos_idx[np.argmin(evals[pos_idx])]
        A, B, C_p, D_p, E, F = evecs[:, idx].real
        if (B**2 - 4*A*C_p) >= 0: return None, None 
    except:
        return None, None

    # 3. Canonical Extraction via Matrix Formulation (100% Robust)
    M = np.array([[A, B/2], [B/2, C_p]])
    M0 = np.array([[A, B/2, D_p/2], 
                   [B/2, C_p, E/2], 
                   [D_p/2, E/2, F]])
    
    try:
        x0_n, y0_n = np.linalg.solve(M, [-D_p/2, -E/2])
    except np.linalg.LinAlgError:
        return None, None
        
    evals_M, evecs_M = np.linalg.eigh(M)
    det_M, det_M0 = np.linalg.det(M), np.linalg.det(M0)
    
    if det_M == 0: return None, None
    K = -det_M0 / det_M
    
    if K / evals_M[0] < 0 or K / evals_M[1] < 0: 
        return None, None # Catch imaginary axes
        
    # evals_M is sorted ascending. Smaller eigenvalue -> Major axis
    a_n = np.sqrt(K / evals_M[0])
    b_n = np.sqrt(K / evals_M[1])
    phi = np.arctan2(evecs_M[1, 0], evecs_M[0, 0])
    
    # 4. Pathological Orthogonal Rejector (Fixes the Cigar Effect)
    cov = np.cov(x_n, y_n)
    evals_cov, evecs_cov = np.linalg.eigh(cov)
    theta_data = np.arctan2(evecs_cov[1, 1], evecs_cov[0, 1]) # Major axis of data density
    
    # Calculate angle difference modulo 180 degrees
    angle_diff = abs((phi - theta_data + np.pi/2) % np.pi - np.pi/2)
    
    # Reject if data is highly elongated AND ellipse is >45 deg off the data's principal axis
    if (evals_cov[1] / max(evals_cov[0], 1e-9) > 3) and (angle_diff > np.pi / 4):
        return None, None
        
    norm_params = {'center': (x0_n, y0_n), 'axes': (a_n, b_n), 'phi': phi}
    norm_info = {'x_min': x_min, 'x_scale': x_scale, 'y_min': y_min, 'y_scale': y_scale}
    return norm_params, norm_info

def _get_geometric_rss(x, y, model_type, params, norm_info=None):
    """Calculates squared Euclidean distance to the curve."""
    data_pts = np.vstack([x, y]).T
    if model_type == 'line':
        m, c = params
        return np.sum(((-m*x + y - c)**2) / (m**2 + 1))
    elif model_type == 'parabola':
        a, b, d = params
        x_p = np.linspace(x.min(), x.max(), 400)
        p_pts = np.vstack([x_p, a*x_p**2 + b*x_p + d]).T
        return np.sum(np.min(np.sum((data_pts[:, None, :] - p_pts[None, :, :])**2, axis=2), axis=1))
    elif model_type == 'ellipse':
        t = np.linspace(0, 2*np.pi, 400)
        a_n, b_n = params['axes']; x0_n, y0_n = params['center']; phi = params['phi']
        x_pts = norm_info['x_min'] + (x0_n + a_n*np.cos(t)*np.cos(phi) - b_n*np.sin(t)*np.sin(phi)) * norm_info['x_scale']
        y_pts = norm_info['y_min'] + (y0_n + a_n*np.cos(t)*np.sin(phi) + b_n*np.sin(t)*np.cos(phi)) * norm_info['y_scale']
        e_pts = np.vstack([x_pts, y_pts]).T
        dists_sq = np.min(np.sum((data_pts[:, None, :] - e_pts[None, :, :])**2, axis=2), axis=1)
        return np.sum(dists_sq), x_pts, y_pts

# def _fit_single_gene_geom(x, y, gene, thresh_p, thresh_e):
#     mask = (x > 0) & (y > 0); x, y = x[mask], y[mask]
#     n = len(x)
#     if n < 10: return None
    
#     # 1. Line
#     lr = linregress(x, y); l_params = (lr.slope, lr.intercept)
#     adj_r2 = 1 - ((1 - lr.rvalue**2) * (n - 1) / (n - 2))
#     rss_l = _get_geometric_rss(x, y, 'line', l_params)
#     bic_l = n * np.log(rss_l / n) + 2 * np.log(n)
    
#     # 2. Parabola
#     p_params = np.linalg.lstsq(np.vstack([x**2, x, np.ones(n)]).T, y, rcond=None)[0]
#     rss_p = _get_geometric_rss(x, y, 'parabola', p_params)
#     bic_p = n * np.log(rss_p / n) + 3 * np.log(n)
    
#     # 3. Ellipse (with Orthogonal Rejector)
#     e_params, norm_info = _fit_ellipse_robust(x, y)
#     bic_e, max_gap = np.inf, 360
#     if e_params:
#         rss_e, _, _ = _get_geometric_rss(x, y, 'ellipse', e_params, norm_info)
#         bic_e = n * np.log(rss_e / n) + 5 * np.log(n)
        
#         dx, dy = (x-norm_info['x_min'])/norm_info['x_scale']-e_params['center'][0], \
#                  (y-norm_info['y_min'])/norm_info['y_scale']-e_params['center'][1]
#         t_data = np.arctan2(dx*np.sin(-e_params['phi'])+dy*np.cos(-e_params['phi']), 
#                             dx*np.cos(-e_params['phi'])-dy*np.sin(-e_params['phi']))
#         t_sorted = np.sort(t_data % (2*np.pi))
#         if len(t_sorted) > 0:
#             max_gap = np.degrees(np.max(np.append(np.diff(t_sorted), 2*np.pi-(t_sorted[-1]-t_sorted[0]))))

#     # --- Pure BIC Selection Logic ---
#     best = "line"
    
#     if bic_p < (bic_l - thresh_p):
#         best = "parabola"
        
#     current_best_bic = bic_p if best == "parabola" else bic_l
#     if bic_e < (current_best_bic - thresh_e):
#         best = "ellipse"
        
#     return {
#         "gene": gene, 
#         "best_fit": best, 
#         "adj_r2_line": adj_r2, 
#         "angular_gap": max_gap, 
#         "branch": "up-regulation" if (best=="parabola" and p_params[0]>0) else "down-regulation" if (best=="parabola") else "steady_state",
#         "coverage": "full cycle" if (best == "ellipse" and max_gap < 120) else "partial"
#     }

# def _fit_single_gene_geom(x, y, gene, thresh_p, thresh_e):
#     mask = (x > 0) & (y > 0); x, y = x[mask], y[mask]
#     n = len(x)
#     if n < 10: return None
    
#     # 1. Normalize data to [0,1] for SCALE-INVARIANT model selection
#     x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
#     x_scale, y_scale = (x_max - x_min) or 1, (y_max - y_min) or 1
#     x_n, y_n = (x - x_min) / x_scale, (y - y_min) / y_scale
#     data_pts_n = np.vstack([x_n, y_n]).T

#     # --- Normalized Line ---
#     lr_n = linregress(x_n, y_n)
#     m_n, c_n = lr_n.slope, lr_n.intercept
#     rss_l_n = np.sum(((-m_n*x_n + y_n - c_n)**2) / (m_n**2 + 1))
#     bic_l = n * np.log(rss_l_n / n) + 2 * np.log(n)
    
#     # We still need unscaled adj_r2 for plotting reference
#     lr_u = linregress(x, y)
#     adj_r2 = 1 - ((1 - lr_u.rvalue**2) * (n - 1) / (n - 2))

#     # --- Normalized Parabola ---
#     p_params_n = np.linalg.lstsq(np.vstack([x_n**2, x_n, np.ones(n)]).T, y_n, rcond=None)[0]
#     # Use 1000 points on normalized [0,1] span for near-perfect continuous approximation
#     x_p = np.linspace(0, 1, 1000) 
#     p_pts_n = np.vstack([x_p, p_params_n[0]*x_p**2 + p_params_n[1]*x_p + p_params_n[2]]).T
#     rss_p_n = np.sum(np.min(np.sum((data_pts_n[:, None, :] - p_pts_n[None, :, :])**2, axis=2), axis=1))
#     bic_p = n * np.log(rss_p_n / n) + 3 * np.log(n)

#     # --- Normalized Ellipse ---
#     e_params, norm_info = _fit_ellipse_robust(x, y)
#     bic_e, max_gap = np.inf, 360
#     if e_params:
#         t = np.linspace(0, 2*np.pi, 1000)
#         a_n, b_n = e_params['axes']; x0_n, y0_n = e_params['center']; phi = e_params['phi']
#         e_x = x0_n + a_n*np.cos(t)*np.cos(phi) - b_n*np.sin(t)*np.sin(phi)
#         e_y = y0_n + a_n*np.cos(t)*np.sin(phi) + b_n*np.sin(t)*np.cos(phi)
#         e_pts_n = np.vstack([e_x, e_y]).T
        
#         rss_e_n = np.sum(np.min(np.sum((data_pts_n[:, None, :] - e_pts_n[None, :, :])**2, axis=2), axis=1))
#         bic_e = n * np.log(rss_e_n / n) + 5 * np.log(n)
        
#         # Calculate Angular Gap for Coverage (using normalized params)
#         dx, dy = x_n - x0_n, y_n - y0_n
#         t_data = np.arctan2(dx*np.sin(-phi) + dy*np.cos(-phi), dx*np.cos(-phi) - dy*np.sin(-phi))
#         t_sorted = np.sort(t_data % (2*np.pi))
#         if len(t_sorted) > 0:
#             max_gap = np.degrees(np.max(np.append(np.diff(t_sorted), 2*np.pi-(t_sorted[-1]-t_sorted[0]))))

#     # --- Selection Logic ---
#     best = "line"
#     if bic_p < (bic_l - thresh_p): best = "parabola"
    
#     current_best_bic = bic_p if best == "parabola" else bic_l
#     if bic_e < (current_best_bic - thresh_e): best = "ellipse"
        
#     # Interpretation uses standard unscaled parameters
#     p_params_u = np.linalg.lstsq(np.vstack([x**2, x, np.ones(n)]).T, y, rcond=None)[0]
    
#     return {
#         "gene": gene, 
#         "best_fit": best, 
#         "adj_r2_line": adj_r2, 
#         "angular_gap": max_gap, 
#         "branch": "up-regulation" if (best=="parabola" and p_params_u[0]>0) else "down-regulation" if (best=="parabola") else "steady_state",
#         "coverage": "full cycle" if (best == "ellipse" and max_gap < 120) else "partial"
#     }

def _fit_single_gene_geom(x_orig, y_orig, gene, thresh_p, thresh_e):
    n_total = len(x_orig)
    # Default state is 0 (steady_state / unassigned)
    cell_states = np.zeros(n_total, dtype=np.int8) 
    
    mask = (x_orig > 0) & (y_orig > 0)
    x, y = x_orig[mask], y_orig[mask]
    n = len(x)
    
    if n < 10: 
        return {"gene": gene, "best_fit": "line", "adj_r2_line": np.nan, 
                "angular_gap": 360, "branch": "steady_state", "coverage": "none", 
                "cell_states": cell_states}
    
    # 1. Normalize data to [0,1] for SCALE-INVARIANT model selection
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    x_scale, y_scale = (x_max - x_min) or 1, (y_max - y_min) or 1
    x_n, y_n = (x - x_min) / x_scale, (y - y_min) / y_scale
    data_pts_n = np.vstack([x_n, y_n]).T

    # --- Normalized Line ---
    lr_n = linregress(x_n, y_n)
    m_n, c_n = lr_n.slope, lr_n.intercept
    rss_l_n = np.sum(((-m_n*x_n + y_n - c_n)**2) / (m_n**2 + 1))
    bic_l = n * np.log(rss_l_n / n) + 2 * np.log(n)
    
    lr_u = linregress(x, y)
    adj_r2 = 1 - ((1 - lr_u.rvalue**2) * (n - 1) / (n - 2))
    
    # ==========================================
    # THE NOISE GATEKEEPER
    # ==========================================
    # 1. Must have a minimum baseline correlation
    # 2. Must not violate Var >= mu biophysics (slope must be positive)
    if adj_r2 < 0.1 or lr_u.slope < 1.0:
        return {
            "gene": gene, 
            "best_fit": "noise", 
            "adj_r2_line": adj_r2, 
            "angular_gap": 360, 
            "branch": "unassigned",
            "coverage": "none",
            "cell_states": np.zeros(n, dtype=np.int8)
        }

    # --- Normalized Parabola ---
    p_params_n = np.linalg.lstsq(np.vstack([x_n**2, x_n, np.ones(n)]).T, y_n, rcond=None)[0]
    x_p = np.linspace(0, 1, 1000) 
    p_pts_n = np.vstack([x_p, p_params_n[0]*x_p**2 + p_params_n[1]*x_p + p_params_n[2]]).T
    rss_p_n = np.sum(np.min(np.sum((data_pts_n[:, None, :] - p_pts_n[None, :, :])**2, axis=2), axis=1))
    bic_p = n * np.log(rss_p_n / n) + 3 * np.log(n)

    # --- Normalized Ellipse ---
    e_params, norm_info = _fit_ellipse_robust(x, y)
    bic_e, max_gap = np.inf, 360
    if e_params:
        phi = e_params['phi']
        # Calculate the slope in the physical (unscaled) domain
        unscaled_slope = np.tan(phi) * (y_scale / x_scale)
        
        t = np.linspace(0, 2*np.pi, 1000)
        a_n, b_n = e_params['axes']; x0_n, y0_n = e_params['center']; phi = e_params['phi']
        e_x = x0_n + a_n*np.cos(t)*np.cos(phi) - b_n*np.sin(t)*np.sin(phi)
        e_y = y0_n + a_n*np.cos(t)*np.sin(phi) + b_n*np.sin(t)*np.cos(phi)
        e_pts_n = np.vstack([e_x, e_y]).T
        
        
        if unscaled_slope < 1.0 or (e_params['axes'][1] / e_params['axes'][0] > 0.8):
            bic_e = np.inf # Disqualify the ellipse
        else:
            rss_e_n = np.sum(np.min(np.sum((data_pts_n[:, None, :] - e_pts_n[None, :, :])**2, axis=2), axis=1))
            bic_e = n * np.log(rss_e_n / n) + 5 * np.log(n)
        
            dx, dy = x_n - x0_n, y_n - y0_n
            t_data = np.arctan2(dx*np.sin(-phi) + dy*np.cos(-phi), dx*np.cos(-phi) - dy*np.sin(-phi))
            t_sorted = np.sort(t_data % (2*np.pi))
            if len(t_sorted) > 0:
                max_gap = np.degrees(np.max(np.append(np.diff(t_sorted), 2*np.pi-(t_sorted[-1]-t_sorted[0]))))

    # --- Selection Logic ---
    best = "line"
    if bic_p < (bic_l - thresh_p): best = "parabola"
    current_best_bic = bic_p if best == "parabola" else bic_l
    if bic_e < (current_best_bic - thresh_e): best = "ellipse"
        
    p_params_u = np.linalg.lstsq(np.vstack([x**2, x, np.ones(n)]).T, y, rcond=None)[0]
    
    # --- Per-Cell Branch Assignment ---
    states_masked = np.zeros(n, dtype=np.int8)
    
    if best == "ellipse":
        x0_n, y0_n = e_params['center']
        phi = e_params['phi']
        
        # Cross product Z-component to determine if point is "above" the major axis
        # y_local > 0 indicates induction (up-regulation)
        y_local = (y_n - y0_n) * np.cos(phi) - (x_n - x0_n) * np.sin(phi)
        
        # THE FIX: Prevent the upside-down eigenvector trap
        if np.cos(phi) < 0:
            y_local = -y_local
        
        states_masked[y_local > 0] = 1   # Up-regulation
        states_masked[y_local < 0] = -1  # Down-regulation
        
    elif best == "parabola":
        # Negative 'a' (concave down) maps to upper arch (1). Positive 'a' maps to lower arch (-1).
        states_masked[:] = 1 if p_params_u[0] < 0 else -1
        
    cell_states[mask] = states_masked
    
    # Gene-level string assignment
    branch_str = "steady_state"
    if best == "parabola":
        branch_str = "up-regulation" if p_params_u[0] < 0 else "down-regulation"
    
    return {
        "gene": gene, 
        "best_fit": best, 
        "adj_r2_line": adj_r2, 
        "angular_gap": max_gap, 
        "branch": branch_str,
        "coverage": "full cycle" if (best == "ellipse" and max_gap < 120) else "partial",
        "cell_states": cell_states
    }

class VelocityModelSelector:
    def __init__(self, adata, mu_layer="mu", var_layer="var"):
        """
        Selector to determine if gene kinetics follow a line, parabola, or ellipse.
        Designed for the sckineticalab package.
        """
        self.adata = adata
        self.mu_layer, self.var_layer = mu_layer, var_layer

#     def run_selection(self, n_jobs=-1, thresh_p=10, thresh_e=15):
#         """
#         Runs model selection using purely Bayesian Information Criterion (BIC).
#         """
#         mu_data, var_data = self.adata.layers[self.mu_layer], self.adata.layers[self.var_layer]
#         if hasattr(mu_data, 'toarray'): mu_data, var_data = mu_data.toarray(), var_data.toarray()
        
#         print(f"Running parallel model selection on {len(self.adata.var_names)} genes...")
#         results = Parallel(n_jobs=n_jobs)(
#             delayed(_fit_single_gene_geom)(mu_data[:, i], var_data[:, i], self.adata.var_names[i], 
#                                           thresh_p, thresh_e) 
#             for i in tqdm(range(len(self.adata.var_names)))
#         )
        
#         res_df = pd.DataFrame([r for r in results if r is not None]).set_index("gene")
#         self.adata.var = self.adata.var.join(res_df, rsuffix='_velo')
#         print("Selection Complete.")

# --- Updated Method in VelocityModelSelector ---
    def run_selection(self, n_jobs=-1, thresh_p=10, thresh_e=15):
        """Runs model selection and assigns cells to kinetic branches."""
        mu_data, var_data = self.adata.layers[self.mu_layer], self.adata.layers[self.var_layer]
        if hasattr(mu_data, 'toarray'): mu_data, var_data = mu_data.toarray(), var_data.toarray()
        
        print(f"Running parallel model selection on {len(self.adata.var_names)} genes...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_gene_geom)(mu_data[:, i], var_data[:, i], self.adata.var_names[i], 
                                          thresh_p, thresh_e) 
            for i in tqdm(range(len(self.adata.var_names)))
        )
        
        # 1. Extract scalar parameters for adata.var
        scalar_results = [{k: v for k, v in r.items() if k != 'cell_states'} for r in results if r is not None]
        res_df = pd.DataFrame(scalar_results).set_index("gene")
        self.adata.var = self.adata.var.join(res_df, rsuffix='_velo')
        
        # 2. Extract cell assignments for adata.layers
        layer_data = np.zeros((self.adata.n_obs, self.adata.n_vars), dtype=np.int8)
        for i, r in enumerate(results):
            if r is not None:
                layer_data[:, i] = r['cell_states']
                
        self.adata.layers["branch_assignment"] = layer_data
        print("Selection Complete. Added 'branch_assignment' to adata.layers.")

    def plot_fits(self, gene):
        """
        Visualizes the three model fits (Linear, Parabolic, Elliptical) for a given gene.
        """
        if gene not in self.adata.var_names:
            print(f"Gene {gene} not found.")
            return
            
        idx = self.adata.var_names.get_loc(gene)
        if isinstance(idx, (slice, np.ndarray, list)): idx = idx[0]
            
        x, y = self.adata.layers[self.mu_layer][:, idx], self.adata.layers[self.var_layer][:, idx]
        mask = (x > 0) & (y > 0); x_f, y_f = x[mask], y[mask]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        # Line
        lr = linregress(x_f, y_f)
        axes[0].scatter(x, y, s=5, alpha=0.3, c='gray')
        axes[0].plot(x_f, lr.slope*x_f+lr.intercept, color='firebrick', lw=2)
        axes[0].set_title(f"Linear (Adj $R^2$: {lr.rvalue**2:.2f})")
        
        # Parabola
        p = np.linalg.lstsq(np.vstack([x_f**2, x_f, np.ones(len(x_f))]).T, y_f, rcond=None)[0]
        axes[1].scatter(x, y, s=5, alpha=0.3, c='gray')
        axes[1].plot(np.sort(x_f), p[0]*np.sort(x_f)**2+p[1]*np.sort(x_f)+p[2], color='forestgreen', lw=2)
        axes[1].set_title(f"Parabolic (a={p[0]:.4f})")
        
        # Ellipse
        axes[2].scatter(x, y, s=5, alpha=0.3, c='gray')
        e_params, norm_info = _fit_ellipse_robust(x_f, y_f)
        
        if e_params:
            _, ex, ey = _get_geometric_rss(x_f, y_f, 'ellipse', e_params, norm_info)
            axes[2].plot(ex, ey, color='royalblue', lw=2)
            axes[2].set_title("Elliptical (Geometric)")
        else: 
            axes[2].set_title("Elliptical\n(Rejected: Orthogonal/Degenerate)")
            
        best = self.adata.var.loc[gene, 'best_fit'] if 'best_fit' in self.adata.var.columns else 'N/A'
        plt.suptitle(f"Model Selection: {gene} | Best: {best}")
        axes[0].set_ylabel(self.var_layer)
        for ax in axes: ax.set_xlabel(self.mu_layer)
        
        plt.tight_layout(); plt.show()