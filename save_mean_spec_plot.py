import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
import glob
from scipy.spatial import ConvexHull


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
    wavelengths = np.linspace(950, 1800, 426)
    foldername_list = ['1000','8020SER','8020PSER','6040SER','6040PSER']  #['1000','9109','9505'] #'6040SER','6040PSER',, '8020SER','8020PSER','9010SER','9010PSER','7030SER','7030PSER','1000'
    sub_folder_list = {'LMT_1','LMT_2'} #,'LMT_2'

    for fl in foldername_list:
        for sub in sub_folder_list:
            folder = f'../res/Caf2_10142025/org/{fl}/{sub}'
            save_subfolder = f'../res/Caf2_10142025/org/mean_spec/'
            os.makedirs(save_subfolder, exist_ok=True)
            # st()
            spectra_all = []
            for file in os.listdir(folder):
                if file.endswith('.npy'):
                    spectrum = np.load(os.path.join(folder, file))
                    _, corrected = rubberband_baseline_correction(wavelengths, spectrum)
                    spectra_all.append(corrected)

            spectra_all = np.array(spectra_all)
            mean_spectrum = np.mean(spectra_all, axis=0)
            std_spectrum = np.std(spectra_all, axis=0)

            # Save mean spectrum as .npy
            np.save(os.path.join(save_subfolder, f'{fl}_{sub}_mean_spectrum.npy'), mean_spectrum)

            # Plot mean ± std
            plt.figure(figsize=(10, 7))
            plt.plot(wavelengths, mean_spectrum, color='blue', label='Mean Spectrum')
            plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                            color='blue', alpha=0.3, label='±1 STD')
            plt.xlabel("Wavelength (cm⁻¹)", fontsize=14)
            plt.ylabel("Normalized Intensity (a.u.)", fontsize=14)
            plt.title(f"{fl} - {sub} Avg Spectrum", fontsize=16)
            plt.legend()
            # plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_subfolder, f'{fl}_{sub}_mean_spectrum.png'))
            plt.close()

if __name__ == '__main__':
    main()