__author__ = 'Tianyi'

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter, find_peaks
from glob import glob
from scipy.ndimage import gaussian_filter1d
from pdb import set_trace as st



def rubberband_baseline_correction(x, y):
    """Rubberband baseline correction using convex hull."""
    x = np.array(x)
    y = np.array(y)

    v = np.vstack((x, y)).T
    hull = ConvexHull(v)

    # Lower hull
    hull_indices = sorted(hull.vertices)
    lower_indices = [idx for idx in hull_indices
                     if idx == 0 or idx == len(x) - 1
                     or (y[idx] < y[idx-1] and y[idx] < y[idx+1])]
    lower_indices = np.array(sorted(lower_indices))

    baseline = np.interp(x, x[lower_indices], y[lower_indices])
    corrected_y = y - baseline
    return baseline, corrected_y


def process_folder(foldername, filename):
    print(f'Processing: {foldername}/{filename}')

    folder_path = f'../spec_res/rat/{foldername}/{filename}/'
    save_path = os.path.join(f'../res/rat_otsu/{foldername}/{filename}/figure')
    os.makedirs(save_path, exist_ok=True)

    wavelengths = np.linspace(950, 1800, 426)

    # ----------------------------
    # Load all spectra
    # ----------------------------
    spec_files = sorted(glob(os.path.join(folder_path, "spec_*.npy")))
    if len(spec_files) == 0:
        print(f"⚠️ No spectra found in {folder_path}")
        return

    spectra = []
    for f in spec_files:
        spec = np.load(f)  # shape should be (426,)
        spectra.append(spec)
    spectra = np.vstack(spectra)  # shape: (N, 426)

    print(f"Loaded {spectra.shape[0]} spectra from {folder_path}")

    # ----------------------------
    # 1. Average and Std
    # ----------------------------
    avg_spectrum = np.mean(spectra, axis=0)
    std_spectrum = np.std(spectra, axis=0)

    # Apply Gaussian smoothing
    avg_spectrum = gaussian_filter1d(avg_spectrum, sigma=2)
    std_spectrum = gaussian_filter1d(std_spectrum, sigma=2)

    _, corrected = rubberband_baseline_correction(wavelengths, avg_spectrum)
    avg_spectrum = corrected

    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, avg_spectrum, color="blue", label="Average Spectrum")
    plt.fill_between(
        wavelengths,
        avg_spectrum - std_spectrum,
        avg_spectrum + std_spectrum,
        color="blue",
        alpha=0.3,
        label="±1 STD",
    )
    plt.xlabel("Wavenumber (cm$^{-1}$)",fontsize=14)
    plt.ylabel("Intensity (a.u.)",fontsize=14)
    plt.title(f"Average Spectrum with STD Shading ({filename})",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename}_avg_std.png"), dpi=300)
    plt.close()

    # ----------------------------
    # 2. 2nd Derivative Spectrum
    # ----------------------------
    avg_spectrum_smooth = savgol_filter(avg_spectrum, window_length=11, polyorder=3)
    second_derivative = savgol_filter(avg_spectrum_smooth, window_length=11, polyorder=3, deriv=2)

    # ----------------------------
    # 3. Sigma in a Range
    # ----------------------------
    range_mask = (wavelengths >= 1280) & (wavelengths <= 1350)
    # sigma_range = np.std(second_derivative[range_mask])
    # st()
    sigma_range = 0.0001
    threshold = 3 * sigma_range

    # Find dips (negative peaks in 2nd derivative)
    peaks_idx, properties = find_peaks(-second_derivative, height=threshold)
    dips_wavelengths = wavelengths[peaks_idx]

    print(f"σ in [1250,1400]: {sigma_range:.4f}")
    print("Dips detected at:", dips_wavelengths)

    # ----------------------------
    # 4. Plot 2nd Derivative + Annotate Dips
    # ----------------------------
    plt.figure(figsize=(8, 3))
    plt.plot(wavelengths, second_derivative, color="red", label="2nd Derivative")
    plt.axhline(-threshold, color="gray", ls="--", label="3σ Threshold (dips)")

    for idx in peaks_idx:
        wn = wavelengths[idx]
        plt.scatter(wn, second_derivative[idx], color="black", s=30)
        plt.text(wn + 10, second_derivative[idx], f"{wn:.0f}", rotation=90, fontsize=12)

    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    plt.ylabel("2nd Derivative (a.u.)", fontsize=16)
    plt.title(f"2nd Derivative Spectrum with 3σ Dips ({filename})", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename}_2nd_derivative_dips.png"), dpi=300)
    plt.close()


def main():
    # Your dataset structure
    foldername_list = ['kidney_ffpe','liver_ffpe'] #'kidney_ff', ' liver_ff',, 'liver_ffpe'
    filename_list = [f'HMT_{i}' for i in range(1, 7)]  # HMT_1 to HMT_6

    for foldername in foldername_list:
        for filename in filename_list:
            process_folder(foldername, filename)


if __name__ == "__main__":
    main()
