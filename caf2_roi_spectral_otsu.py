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

    foldername_list = ['1000/'] #,'9010/','8020/','7030/','6040/'
    filename_list = ['LMT_1','LMT_2','LMT_3']
    for foldername in foldername_list:
        for filename in filename_list:
            print('processing:', foldername, filename)
            after_collagen = r'W:/3. Students/Tianyi/Caf2_09022025/'+f'{foldername}'+f'{filename}'+'.mat'
            save_path = f'C:/pyws/SPEC/res/Caf2_09022025/{foldername}'
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
            ax4.imshow(binary_mask_region0, cmap='gray')
            ax4.set_title('Binary Mask - Region 0')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{filename}_spatial_image.png'))
            # plt.show()
            # st()
            

            mean_spectrum_after = np.mean(binary_mask_region0.T[:, :, np.newaxis]*spectra_after, axis=(0, 1))
            std_spectrum_after = np.std(binary_mask_region0.T[:, :, np.newaxis]*spectra_after, axis=(0, 1))
            mean_spectrum_after_roi = np.mean(binary_mask_region1.T[:, :, np.newaxis]*spectra_after, axis=(0, 1))
            std_spectrum_after_roi = np.std(binary_mask_region1.T[:, :, np.newaxis]*spectra_after, axis=(0, 1))

            # Apply aALS baseline correction
            # lam = 1e6  # Adjust as needed
            # p = 0.01   # Adjust as needed
            # baseline_before, corrected_spectrum_before = als_baseline_correction(mean_spectrum_after, lam=lam, p=p)
            # baseline_after, corrected_spectrum_after = als_baseline_correction(mean_spectrum_after_roi, lam=lam, p=p)

            baseline_before, corrected_spectrum_before = rubberband_baseline_correction(wavelengths, mean_spectrum_after)
            baseline_after, corrected_spectrum_after = rubberband_baseline_correction(wavelengths, mean_spectrum_after_roi)

            # Apply Gaussian smoothing
            sigma = 1  # Adjust smoothing level; try 1-3
            mean_spectrum_after = gaussian_filter1d(mean_spectrum_after, sigma=sigma)
            mean_spectrum_after_roi = gaussian_filter1d(mean_spectrum_after_roi, sigma=sigma)
            corrected_spectrum_before = gaussian_filter1d(corrected_spectrum_before, sigma=sigma)
            corrected_spectrum_after = gaussian_filter1d(corrected_spectrum_after, sigma=sigma)

            # Plot the average spectrum with standard deviation
            plt.figure(figsize=(12, 8))
            # plt.plot(wavelengths, mean_spectrum_after, label='Smoothed Average Spectrum - Mask 0', color='b', linewidth=2) 
            # plt.plot(wavelengths, mean_spectrum_after_roi, label='Smoothed Average Spectrum - Mask 1', color='b', linewidth=2) 
            plt.plot(wavelengths, corrected_spectrum_before, label='Region 0', color='k', linewidth=3) 
            plt.plot(wavelengths, corrected_spectrum_after, label='Region 1', color='r', linewidth=3)
            # plt.fill_between(wavelengths, 
            #                 mean_spectrum_after - std_spectrum_after, 
            #                 mean_spectrum_after + std_spectrum_after, 
            #                 color='b', alpha=0.3, label='Standard Deviation')
            # plt.fill_between(wavelengths, 
            #                 mean_spectrum_after_roi - std_spectrum_after_roi, 
            #                 mean_spectrum_after_roi + std_spectrum_after_roi, 
            #                 color='b', alpha=0.3, label='Standard Deviation')
            plt.fill_between(wavelengths, 
                            corrected_spectrum_before - std_spectrum_after, 
                            corrected_spectrum_before + std_spectrum_after, 
                            color='k', alpha=0.3, label='Standard Deviation')
            plt.fill_between(wavelengths, 
                            corrected_spectrum_after - std_spectrum_after_roi, 
                            corrected_spectrum_after + std_spectrum_after_roi, 
                            color='r', alpha=0.3, label='Standard Deviation') 
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Intensity')
            plt.legend(loc='upper left', fontsize=14)
            # plt.grid(True)
            plt.savefig(os.path.join(save_path, f'average_spectrum_{filename}.png'))
            # plt.show()

        

            # # Save the subspectrum
            # # subspectrum_filename = f"{filename}_after_mask0.mat"
            # # savemat(os.path.join(save_path, subspectrum_filename), {'spectrum': corrected_spectrum_before})
            subspectrum_filename_1 = f"{filename}_after_mask1.mat"
            savemat(os.path.join(save_path, subspectrum_filename_1), {'spectrum': corrected_spectrum_after})
            print(f'Subspectrum saved!')



if __name__ == '__main__':
    main()
