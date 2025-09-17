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

    foldername_list = ['Caf2_03132025_rat_ffpe/liver_ffpe/'] #Caf2_03072025_rat_oct/liver_oct  #Caf2_03132025_rat_ffpe/liver_ffpe/
    filename_list = ['HMT_10','HMT_5','HMT_4','HMT_3','HMT_2','HMT_7','HMT_9'] #'HMT_10','HMT_5','HMT_4','HMT_3',
    for foldername in foldername_list:
        for filename in filename_list:
            print('processing:', foldername, filename)
            after_collagen = r'D:/Sperodata/'+f'{foldername}'+f'{filename}'+'.mat'
            save_path = f'D:/spec_res/rat/{foldername}'
            os.makedirs(save_path, exist_ok=True)


            wavelengths = np.linspace(950, 1800, 426)
            data_after = loadmat(after_collagen)
            spectra_after = np.reshape(data_after['r'], (480, 480, 426))

            mean_spectrum = np.mean(spectra_after, axis=(0, 1))
            std_spectrum = np.std(spectra_after, axis=(0, 1))

            fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(9, 6))  # Create a 2x2 grid of subplots
            ax1.imshow(spectra_after[:,:,330].T)
            ax1.set_title(f'{filename}_spectra')
            thresholds = threshold_multiotsu(spectra_after[:, :, 330].T, classes=3)
            # print(thresholds) 
            # thresholds = np.array([0.05, 0.1, 0.8])
            regions = np.digitize(spectra_after[:, :, 330].T, bins=thresholds)
            # st()
            binary_mask_region0 = (regions == 0).astype(np.uint8)
            binary_mask_region1 = (regions == 1).astype(np.uint8)
            binary_mask_region2 = (regions == 2).astype(np.uint8)
            ax2.imshow(binary_mask_region1, cmap='gray')
            ax2.set_title('Binary Mask - Region 1')
            ax3.imshow(binary_mask_region0, cmap='gray')
            ax3.set_title('Binary Mask - Region 0')
            ax4.imshow(binary_mask_region2, cmap='gray')
            ax4.set_title('Binary Mask - Region 2')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{filename}_spatial_image.png'))
            # plt.show()
            # st()
            data = binary_mask_region2.T[:, :, np.newaxis]*spectra_after

            # Apply mask to get spectra only from region 2 (high intensity)
            data = binary_mask_region2.T[:, :, np.newaxis] * spectra_after  # shape (480, 480, 426)

            # Reshape to list of all pixels: (N_pixels, N_wavenumbers)
            flattened_data = data.reshape(-1, data.shape[2])  # (480*480, 426)

            # Only keep non-zero spectra (non-masked areas will be all zeros)
            nonzero_indices = np.where(np.any(flattened_data != 0, axis=1))[0]
            print(f'Total valid spectra in region 2: {len(nonzero_indices)}')

            # Randomly sample 2000 spectra (or fewer if not enough)
            n_samples = min(2000, len(nonzero_indices))
            selected_indices = np.random.choice(nonzero_indices, size=n_samples, replace=False)
            selected_spectra = flattened_data[selected_indices]  # shape (n_samples, 426)

            # Create subfolder for .npy files if needed
            npy_dir = os.path.join(save_path, f'{filename}_mask2_corrected_spectra')
            os.makedirs(npy_dir, exist_ok=True)

            # Loop and save each corrected spectrum
            for i in range(n_samples):
                _, corrected = rubberband_baseline_correction(wavelengths, selected_spectra[i])
                np.save(os.path.join(npy_dir, f'spec_{i:04d}.npy'), corrected)

            print(f'{n_samples} corrected .npy spectra saved to {npy_dir}')




if __name__ == '__main__':
    main()
