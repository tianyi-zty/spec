import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
import glob
from scipy.spatial import ConvexHull


def amideI_normalize(wn, y, lo=1600, hi=1700):
    wn = np.asarray(wn).ravel()
    y = np.asarray(y).ravel()

    mask = (wn >= lo) & (wn <= hi)
    if not np.any(mask):
        raise ValueError(f"No points in Amide I window {lo}-{hi} cm^-1. Check wavenumber array.")
    peak = np.max(y[mask])
    if peak == 0:
        raise ValueError("Amide I peak is 0, cannot normalize.")
    return y / peak, peak


def rubberband_baseline_correction(x, y):
    """
    Rubberband baseline correction using the convex hull.
    
    Parameters:
        x (array-like): The x-axis values (e.g., wavenumber).
        y (array-like): The y-axis values (e.g., intensity).

    Returns:
        baseline (array): The rubberband baseline.
        corrected_y (array): The baseline-corrected spectrum.
    """
    x = np.array(x)
    y = np.array(y)

    # Get points forming the convex hull
    v = np.vstack((x, y)).T
    hull = ConvexHull(v)

    # Extract lower convex hull indices (start and end inclusive)
    hull_indices = sorted(hull.vertices)
    lower_indices = [idx for idx in hull_indices if idx == 0 or idx == len(x) - 1 or (y[idx] < y[idx-1] and y[idx] < y[idx+1])]
    lower_indices = np.array(sorted(lower_indices))

    # Interpolate baseline across those points
    baseline = np.interp(x, x[lower_indices], y[lower_indices])

    corrected_y = y - baseline

    return baseline, corrected_y

# # Final absorbance spectrum (sum + baseline)
# total_abs = abs_lipids + abs_proteins + abs_nucleic + abs_carb + baseline
wavelengths = np.linspace(950, 1800, 426)
path_3 = r'C:/pyws/SPEC/res/02192026-aca'
# spectra_col = np.load(os.path.join(path_3, "COL4-1_after_mask1.mat"))
col_data = loadmat(os.path.join(path_3, "02Maca-1_after_mask1.mat"))
spectra_col = np.reshape(col_data['spectrum'], (426))
baseline,spectra_col_als = rubberband_baseline_correction(wavelengths,spectra_col)

# normalize so Amide I peak = 1 (use baseline-corrected spectrum)
spectra_col_als_norm, col4_peak = amideI_normalize(wavelengths, spectra_col_als, lo=1600, hi=1700)

path_4 = r'C:/pyws/SPEC/res/02192026-aca'
# spectra_col_4 = np.load(os.path.join(path_4, "COL1_after_mask2.ny"))
col_data_4 = loadmat(os.path.join(path_4, "5Maca-2_after_mask1.mat"))
spectra_col_4 = np.reshape(col_data_4['spectrum'], (426))
baseline_1,spectra_col_als_4 = rubberband_baseline_correction(wavelengths,spectra_col_4)

# normalize so Amide I peak = 1 (use baseline-corrected spectrum)
spectra_col_als_4_norm, col1_peak = amideI_normalize(wavelengths, spectra_col_als_4, lo=1000, hi=1200)

print(f"COL4 Amide I peak (pre-norm): {col4_peak:.4f}")
print(f"COL1 Amide I peak (pre-norm): {col1_peak:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
# Plot final spectrum
plt.plot(wavelengths, spectra_col_als, color='red', linewidth=3, label='0.02M AcA')
plt.plot(wavelengths, spectra_col_als_4+0.003, color='blue', linewidth=3, label='0.5M AcA')

# Labels and formatting
plt.xlabel('Wavenumber (cm⁻¹)',fontsize=18)
plt.ylabel('Absorbance (a.u.)',fontsize=18)
# plt.title('Illustrative Absorbance Spectrum')
plt.legend(loc='upper left', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim((-0.002,0.012))
# plt.grid(True)
plt.tight_layout()
plt.show()
