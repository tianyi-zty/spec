import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
from skimage.filters import threshold_multiotsu
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter1d


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

    foldername_list = ['kidney_ffpe/'] #Caf2_03072025_rat_oct/liver_oct  #Caf2_03132025_rat_ffpe/liver_ffpe/
    filename_list = ['HMT_1'] #'HMT_10','HMT_5','HMT_4','HMT_3','HMT_5','HMT_4','HMT_3','HMT_2','HMT_7',
    for foldername in foldername_list:
        for filename in filename_list:
            print('processing:', foldername, filename)
            after_collagen = r'/Volumes/TIANYI/Sperodata/Caf2_03132025_rat_ffpe/'+f'{foldername}'+f'{filename}'+'.mat'
            save_path = f'/Volumes/TIANYI/spec_res/rat/{foldername}'
            os.makedirs(save_path, exist_ok=True)


            wavelengths = np.linspace(950, 1800, 426)
            data_after = loadmat(after_collagen)
            spectra_after = np.reshape(data_after['r'], (480, 480, 426))

            mean_spectrum = np.mean(spectra_after, axis=(0, 1))
            std_spectrum = np.std(spectra_after, axis=(0, 1))

            thresholds = threshold_multiotsu(spectra_after[:, :, 330].T, classes=3)
            regions = np.digitize(spectra_after[:, :, 330].T, bins=thresholds)
            binary_mask_region0 = (regions == 0).astype(np.uint8)
            binary_mask_region1 = (regions == 1).astype(np.uint8)
            binary_mask_region2 = (regions == 2).astype(np.uint8)
            data = binary_mask_region0.T[:, :, np.newaxis]*spectra_after

            # Reshape to list of all pixels: (N_pixels, N_wavenumbers)
            flattened_data = data.reshape(-1, data.shape[2])  # (480*480, 426)
            st()
            mean_spectrum = np.mean(flattened_data, axis=0)

            # Create subfolder for .npy files if needed
            npy_dir = os.path.join(save_path, f'{filename}_bg')
            os.makedirs(npy_dir, exist_ok=True)
            _, corrected = rubberband_baseline_correction(wavelengths, mean_spectrum)
            np.save(os.path.join(npy_dir, f'bg.npy'), corrected)

            print(f'bg.npy spectrum saved to {npy_dir}')




if __name__ == '__main__':
    main()
