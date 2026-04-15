__author__ = 'Tianyi'

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, find_peaks


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


def main():

    foldername_list = ['1000','8020SER','8020PSER','6040SER','6040PSER'] #['kidney_oct'] #Caf2_03072025_rat_oct/liver_oct  #Caf2_03132025_rat_ffpe/liver_ffpe/
    filename_list = ['LMT_1', 'LMT_2'] #'HMT_10','HMT_5','HMT_4','HMT_3',
    # cluster = ['cluster_0','cluster_1']
    #######load the spectrum after tsne filtering tsne_filter_save.py#########
    for foldername in foldername_list:
        for filename in filename_list:
            # for cluster in cluster:
                print('processing:', foldername, filename)
                folder_path = f'../res/Caf2_10142025/org/{foldername}/{filename}'
                save_path = f'../res/Caf2_10142025/3sigma_peaks/figure'
                os.makedirs(save_path, exist_ok=True)

                wavelengths = np.linspace(950, 1800, 426)
                # data_after = loadmat(data)
                # spectra_after = np.reshape(data_after['r'], (480, 480, 426))
                # spectrum = np.load(os.path.join(data))
                spec_files = sorted(glob(os.path.join(folder_path, "*.npy")))
                if len(spec_files) == 0:
                    print(f"⚠️ No spectra in {folder_path}")
                    return

                np.random.seed(42)  
                chosen_files = np.random.choice(spec_files, size=10, replace=False)
                results = []

                # ----------------------------
                # Process each spectrum
                # ----------------------------
                n=0
                for spec_file in chosen_files:
                    spec = np.load(spec_file)
                    n += 1
                    print(f"Processing spectrum {n}...")    
                    # Smooth + Baseline correction
                    # spec_smooth = gaussian_filter1d(spec, sigma=3)
                    spec_smooth = spec
                    _, spec_corrected = rubberband_baseline_correction(wavelengths, spec_smooth)

                    # 2nd derivative
                    smooth = savgol_filter(spec_corrected, window_length=11, polyorder=3)
                    second_derivative = savgol_filter(smooth, window_length=11, polyorder=3, deriv=2)

                    # 3σ threshold
                    threshold = 0.0005

                    peaks_idx, _ = find_peaks(-second_derivative, height=threshold)
                    dips_wavelengths = wavelengths[peaks_idx]

                    # ----------------------------
                    # 4. Plot 2nd Derivative + Annotate Dips
                    # ----------------------------
                    plt.figure(figsize=(12, 3))
                    plt.plot(wavelengths, second_derivative, color="black", label="2nd Derivative")
                    plt.axhline(-threshold, color="gray", ls="--", label="3σ Threshold (dips)")

                    # Mark dips
                    for idx in peaks_idx:
                        wn = wavelengths[idx]
                        plt.scatter(wn, second_derivative[idx], color="red", s=60)
                        plt.text(wn+4, second_derivative[idx], f"{wn:.1f}", rotation=90, fontsize=14)

                    plt.xlabel("Wavenumber (cm$^{-1}$)")
                    plt.ylabel("2nd Derivative (a.u.)")
                    # plt.title("2nd Derivative Spectrum with 3σ Dip Detection")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f"{foldername}_{filename}_{n}.png"), dpi=300)
                    # plt.show()
                    plt.close()



if __name__ == "__main__":
    main()