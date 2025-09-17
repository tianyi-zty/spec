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

# def gaussian(x, mu, sigma, amplitude):
#     return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

# # Wavenumber range from 950 to 1800 cm⁻¹ (left to right)
# wavenumbers = np.linspace(950, 1800, 1000)[::-1]  # flip axis direction

# # Create peaks for each biomolecule
# abs_lipids = gaussian(wavenumbers, 1740, 20, 0.2) 
# abs_proteins = gaussian(wavenumbers, 1650, 25, 1.3) + gaussian(wavenumbers, 1550, 25, 1.0)
# abs_nucleic = gaussian(wavenumbers, 1230, 10, 0.5) + gaussian(wavenumbers, 1080, 20, 0.4)
# abs_carb = gaussian(wavenumbers, 1050, 10, 0.3)

# # Random baseline (low-frequency sine + random noise)
# np.random.seed(42)  # for reproducibility
# baseline = 0.1 * np.sin(np.linspace(0, 2*np.pi, len(wavenumbers))) + 0.01 * np.random.randn(len(wavenumbers))
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
path_2 = r'../res/rat/kidney_ffpe/'
col_data_2 = loadmat(os.path.join(path_2, "HMT_5_after_mask1.mat"))
spectra_col_2 = np.reshape(col_data_2['spectrum'], (426))
baseline_2, spectra_col_als_2 = rubberband_baseline_correction(wavelengths,spectra_col_2)

path_4 = r'../res/rat/kidney_ff/'
col_data_4 = loadmat(os.path.join(path_4, "HMT_5_after_mask1.mat"))
spectra_col_4 = np.reshape(col_data_4['spectrum'], (426))
baseline_1,spectra_col_als_4 = rubberband_baseline_correction(wavelengths,spectra_col_4)


# Plotting
plt.figure(figsize=(10, 6))

# Colored background for each biomolecule region
plt.axvspan(1720, 1755, color='purple', alpha=0.1, label='Lipids')
plt.axvspan(1610, 1690, color='green', alpha=0.1, label='Proteins')
plt.axvspan(1500, 1600, color='green', alpha=0.1)
plt.axvspan(1215, 1245, color='blue', alpha=0.1, label='Nucleic Acids')
plt.axvspan(1065, 1095, color='blue', alpha=0.1)
plt.axvspan(1020, 1120, color='orange', alpha=0.1, label='Carbohydrates')

# Plot final spectrum
plt.plot(wavelengths, spectra_col_als_2, color='orange', linewidth=3, label='Kidney_ffpe')
plt.plot(wavelengths, spectra_col_als_4, color='green', linewidth=3, label='Kidney_ff')

# Labels and formatting
plt.xlabel('Wavenumber (cm⁻¹)',fontsize=18)
plt.ylabel('Absorbance (a.u.)',fontsize=18)
# plt.title('Illustrative Absorbance Spectrum')
plt.legend(loc='upper left', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.grid(True)
plt.tight_layout()
plt.show()
