# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from pdb import set_trace as st

# ----------------------------
# Paths (yours)
# ----------------------------
col1_mat   = r"C:/pyws/SPEC/res/Caf2_02162026/col1/COL1_after_mask2.mat"
col4_mat   = r"C:/pyws/SPEC/res/Caf2_02162026/col4/COL4-1_after_mask1.mat"
aca05_mat  = r"C:/pyws/SPEC/res/02192026-aca/5Maca-2_after_mask1.mat"
aca002_mat = r"C:/pyws/SPEC/res/02192026-aca/02Maca-1_after_mask1.mat"

# ----------------------------
# Config
# ----------------------------
FIT_RANGE   = (900, 1800)   # cm^-1
POLY_ORDER  = 2
NONNEG_ACID = True
AMIDE_RANGE = (1600, 1700)

# ----------------------------
# Load spectrum from .mat
# ----------------------------
def load_spectrum(mat_path, key="spectrum"):
    d = loadmat(mat_path)
    if key not in d:
        raise KeyError(f"'{key}' not found in {mat_path}. Keys: {list(d.keys())}")
    y = np.asarray(d[key]).squeeze().astype(float)
    # Force to (m,n)
    if y.ndim == 1:
        Y = y[None, :]
    elif y.ndim == 2:
        Y = y.astype(float)
    else:
        Y = y.reshape(-1, y.shape[-1]).astype(float)
    return Y

# ----------------------------
# Build wn axis from endpoints
# ----------------------------
def make_wn_from_endpoints(Y, lo=900, hi=1800, descending=False):
    """
    Create a wavenumber axis with length matching Y.shape[1].
    """
    n = Y.shape[1]
    wn = np.linspace(lo, hi, n)
    if descending:
        wn = wn[::-1]
    return wn

# ----------------------------
# EMSC correction
# ----------------------------
def emsc_acid_correct(wn, Y, acid_ref, target=None,
                      poly_order=2, fit_range=(900, 1800),
                      nonneg_acid=True):
    """
    y = poly(wn) + c*target + a*acid_ref + eps
    y_corr = (y - poly - a*acid_ref) / c
    """
    wn = np.asarray(wn, float).ravel()
    Y = np.asarray(Y, float)
    acid_ref = np.asarray(acid_ref, float).ravel()

    if target is None:
        target = np.nanmean(Y, axis=0)
    target = np.asarray(target, float).ravel()

    lo, hi = fit_range
    idx = (wn >= lo) & (wn <= hi)
    if not np.any(idx):
        raise ValueError("fit_range gives empty mask. Check wn.")

    # numeric stability
    v = (wn - wn.mean()) / (wn.std() + 1e-12)
    v_fit = v[idx]

    # Design matrix: [1, v, v^2, ..., target, acid]
    cols = [np.ones_like(v_fit)]
    for k in range(1, poly_order + 1):
        cols.append(v_fit ** k)
    cols.append(target[idx])
    cols.append(acid_ref[idx])
    X = np.vstack(cols).T  # (n_fit, p)

    coeffs, *_ = np.linalg.lstsq(X, Y[:, idx].T, rcond=None)
    coeffs = coeffs.T

    p_poly = poly_order + 1
    c = coeffs[:, p_poly]      # target scale
    a = coeffs[:, p_poly + 1]  # acid scale
    if nonneg_acid:
        a = np.maximum(a, 0.0)

    # polynomial over full wn
    cols_full = [np.ones_like(v)]
    for k in range(1, poly_order + 1):
        cols_full.append(v ** k)
    P = np.vstack(cols_full)           # (p_poly, n)
    poly = coeffs[:, :p_poly] @ P      # (m, n)

    acid_term = np.outer(a, acid_ref)  # (m, n)

    c_safe = np.where(np.abs(c) < 1e-12, 1.0, c)
    Ycorr = (Y - poly - acid_term) / c_safe[:, None]

    return Ycorr, {"acid_scale_a": a, "target_scale_c": c, "coeffs": coeffs}

def normalize_amide1_peak(wn, Y, amide_range=(1600, 1700)):
    wn = np.asarray(wn, float).ravel()
    Y = np.asarray(Y, float)
    mask = (wn >= amide_range[0]) & (wn <= amide_range[1])
    if not np.any(mask):
        raise ValueError("Amide I mask empty; check wn.")
    s = np.nanmax(Y[:, mask], axis=1)
    s_safe = np.where(np.abs(s) < 1e-12, 1.0, s)
    return Y / s_safe[:, None]

# ----------------------------
# Main
# ----------------------------
col1 = load_spectrum(col1_mat)
col4 = load_spectrum(col4_mat)
Y05  = load_spectrum(aca05_mat)
Y002 = load_spectrum(aca002_mat)

# Build wn axis from endpoints (length = number of spectral points)
wn = make_wn_from_endpoints(col1, lo=FIT_RANGE[0], hi=FIT_RANGE[1], descending=False)

# Sanity check lengths
n = wn.size
for name, arr in [("col1", col1), ("col4", col4), ("aca05", Y05), ("aca002", Y002)]:
    if arr.shape[1] != n:
        raise ValueError(f"{name} has {arr.shape[1]} points but wn has {n}.")

acid05  = np.nanmean(Y05, axis=0).ravel()
acid002 = np.nanmean(Y002, axis=0).ravel()

# Match solvent blank to sample:
# Col I ~0.02M AcA 
# Col IV ~0.5M AcA 
col1_corr, p1 = emsc_acid_correct(wn, col1, acid002, poly_order=POLY_ORDER, fit_range=FIT_RANGE, nonneg_acid=NONNEG_ACID)
col4_corr, p4 = emsc_acid_correct(wn, col4, acid05,  poly_order=POLY_ORDER, fit_range=FIT_RANGE, nonneg_acid=NONNEG_ACID)
st()
print(np.mean(p1["acid_scale_a"]), np.mean(p4["acid_scale_a"]))

# Normalize (optional)
col1_corr_n = normalize_amide1_peak(wn, col1_corr, amide_range=AMIDE_RANGE)
col4_corr_n = normalize_amide1_peak(wn, col4_corr, amide_range=AMIDE_RANGE)
col1_n = normalize_amide1_peak(wn, col1, amide_range=AMIDE_RANGE)
col4_n = normalize_amide1_peak(wn, col4, amide_range=AMIDE_RANGE)

# Save
out_dir = 'C:/pyws/SPEC/res/Caf2_02162026/'
savemat(os.path.join(out_dir, "col1_emsc_acid_corrected.mat"), {
    "wavenumbers": wn,
    "col1_corr": col1_corr,
    "col1_corr_norm": col1_corr_n,
    "acid_scale_a": p1["acid_scale_a"],
    "target_scale_c": p1["target_scale_c"],
})
savemat(os.path.join(out_dir, "col4_emsc_acid_corrected.mat"), {
    "wavenumbers": wn,
    "col4_corr": col4_corr,
    "col4_corr_norm": col4_corr_n,
    "acid_scale_a": p4["acid_scale_a"],
    "target_scale_c": p4["target_scale_c"],
})

print("Saved corrected .mat files to:", out_dir)

# Plot mean corrected spectra in FIT_RANGE
mask_plot = (wn >= FIT_RANGE[0]) & (wn <= FIT_RANGE[1])
col1_plot = np.nanmean(col1_corr_n, axis=0)
col4_plot = np.nanmean(col4_corr_n, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(wn[mask_plot], col1_plot[mask_plot], linewidth=3, label="Col I (corr, AmideI norm)")
plt.plot(wn[mask_plot], col4_plot[mask_plot]+0.2, linewidth=3, label="Col IV (corr, AmideI norm)")
# plt.plot(wn, np.nanmean(col4_corr - col4_n[0, :], axis=0))

# plt.plot(wn[mask_plot], col1_n[0, :]+0.4, linewidth=3, label="Col I (AmideI norm)")
# plt.plot(wn[mask_plot], col4_n[0, :]+0.6, linewidth=3, label="Col IV (AmideI norm)")
plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=18)
plt.ylabel("Absorbance (a.u.)", fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.tight_layout()
plt.show()
