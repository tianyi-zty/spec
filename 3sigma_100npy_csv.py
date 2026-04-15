__author__ = 'Tianyi'

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
from glob import glob
import pandas as pd


def rubberband_baseline_correction(x, y):
    """Rubberband baseline correction using convex hull."""
    v = np.vstack((x, y)).T
    hull = ConvexHull(v)
    hull_indices = sorted(hull.vertices)

    lower = [idx for idx in hull_indices
             if idx == 0 or idx == len(x) - 1
             or (y[idx] < y[idx - 1] and y[idx] < y[idx + 1])]
    lower = np.array(sorted(lower))

    baseline = np.interp(x, x[lower], y[lower])
    corrected_y = y - baseline
    return baseline, corrected_y


def process_folder(foldername, filename):
    print(f'Processing: {foldername}/{filename}')

    folder_path = f'../res/Caf2_10142025/org/{foldername}/{filename}/'
    save_path = f'../res/Caf2_10142025/3sigma_peaks/{foldername}/'
    os.makedirs(save_path, exist_ok=True)

    wavelengths = np.linspace(950, 1800, 426)

    # ----------------------------
    # Load spectra (random 100)
    # ----------------------------
    spec_files = sorted(glob(os.path.join(folder_path, "*.npy")))
    if len(spec_files) == 0:
        print(f"⚠️ No spectra in {folder_path}")
        return

    np.random.seed(42)  # reproducibility
    chosen_files = np.random.choice(spec_files, size=100, replace=False)

    results = []

    # ----------------------------
    # Process each spectrum
    # ----------------------------
    for spec_file in chosen_files:
        spec = np.load(spec_file)

        # Smooth + Baseline correction
        # spec_smooth = gaussian_filter1d(spec, sigma=3)
        spec_smooth = spec
        _, spec_corrected = rubberband_baseline_correction(wavelengths, spec_smooth)

        # 2nd derivative
        smooth = savgol_filter(spec_corrected, window_length=11, polyorder=3)
        second_derivative = savgol_filter(smooth, window_length=11, polyorder=3, deriv=2)

        # 3σ threshold
        threshold = 0.0003

        peaks_idx, _ = find_peaks(-second_derivative, height=threshold)
        dips_wavelengths = wavelengths[peaks_idx]

        # Save as a row (filename + list of dips)
        results.append({
            "spectrum_file": os.path.basename(spec_file),
            "num_dips": len(dips_wavelengths),
            "dips_wavenumber": ";".join([f"{wn:.2f}" for wn in dips_wavelengths])
        })

    # ----------------------------
    # Save all 100 spectra results to CSV
    # ----------------------------
    csv_path = os.path.join(save_path, f"{filename}_100spectra_dips.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")


def main():
    foldername_list = ['1000','8020SER','8020PSER','6040SER','6040PSER']  # add more if needed
    filename_list = [f'LMT_{i}' for i in range(1, 2)]  # LMT_1 to LMT_6

    for foldername in foldername_list:
        for filename in filename_list:
            process_folder(foldername, filename)


if __name__ == "__main__":
    main()
