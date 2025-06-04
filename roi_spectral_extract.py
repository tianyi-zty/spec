import os
from scipy.io import loadmat
from skimage.filters import threshold_otsu
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st

# in this code, i want to first visualize the spero data as an image at 1600cm-1[330;170;260;70] (change according to each resonance), then, generate a mask according to the dark pixel
# then, only plot the spectrum of the mask=1 (assume this is pure collagen pixel data).
# save the spectrum and plot
# switch to 2nd_derivateive_caf2.py detect peaks
# then need a code to plot the spectrum and 2nd derivate of all data together
def save_spectrum_to_mat(spectrum, filename, save_path):
    """
    Save the spectrum to a .mat file at the specified path.
    
    Parameters:
        spectrum (array-like): The spectrum to save.
        filename (str): The name of the output .mat file.
        save_path (str): The directory path where the .mat file should be saved.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the directory if it doesn't exist
    
    full_path = os.path.join(save_path, filename)
    data_dict = {'corrected_spectrum': spectrum}
    savemat(full_path, data_dict)
    print(f"Spectrum saved to {full_path}")


def main():
    filename = f'8'
    before_collagen = r'/Volumes/TIANYI/Sperodata/AuPillars_50nmAl2O3_6_05232025/before/LMR_2.mat'
    after_collagen = r'/Volumes/TIANYI/Sperodata/AuPillars_50nmAl2O3_6_05232025/after/LMR_3.mat'
    save_path = f'../res/peptide1/AuPillars_50nmAl2O3_6_05232025/{filename}'
    os.makedirs(save_path, exist_ok=True)

    wavelengths = np.linspace(950, 1800, 426)
    data_before = loadmat(before_collagen)
    spectra_before = np.reshape(data_before['r'], (480, 480, 426))
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))

    x_start_1, x_end_1 = 310,474
    y_start_1, y_end_1 = 318,452
    region_before = spectra_before[x_start_1:x_end_1, y_start_1:y_end_1, :]
    x_start_2, x_end_2 = 276,424
    y_start_2, y_end_2 = 333,474
    region_after = spectra_after[x_start_2:x_end_2, y_start_2:y_end_2, :]

    n=60
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 6))  # Create a 2x2 grid of subplots
    ax1.imshow(spectra_before[:,:,n].T)
    ax1.set_title('spectra_before')
    ax2.imshow(region_before[:,:,n].T)
    ax2.set_title('extracted_region_before')
    ax4.imshow(spectra_after[:,:,n].T)
    ax4.set_title('spectra_after')
    ax5.imshow(region_after[:,:,n].T)
    ax5.set_title('extracted_region_after')
    # Step 1: Compute Otsu's threshold
    otsu_thresh = threshold_otsu(region_after[:,:,n].T)-0.07
    print(otsu_thresh)
    # Step 2: Create binary mask using the computed threshold
    binary_mask = (region_after[:,:,n].T < otsu_thresh).astype(np.uint8)
    ax6.imshow(binary_mask, cmap='gray')
    ax6.set_title('Binary Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{filename}_spatial_image.png'))
    plt.show()
    st()

    mean_spectrum_before = np.mean(region_before, axis=(0, 1))
    mean_spectrum_after = np.mean(region_after, axis=(0, 1))
    mean_spectrum_after_roi = np.mean(binary_mask.T[:, :, np.newaxis]*region_after, axis=(0, 1))
    mean_spectrum_after_roi_1 = np.mean((1-binary_mask).T[:, :, np.newaxis]*region_after, axis=(0, 1))
    # st()
    # Calculate 10^(-mean_spectrum_after)
    transformed_spectrum_before = 10 ** (-mean_spectrum_before)
    transformed_spectrum_after = 10 ** (-mean_spectrum_after)
    transformed_spectrum_after_roi = 10 ** (-mean_spectrum_after_roi)
    transformed_spectrum_after_roi_1 = 10 ** (-mean_spectrum_after_roi_1)

    # Save the subspectrum
    subspectrum_filename = f"resonance_{filename}_spectrum_before.mat"
    savemat(os.path.join(save_path, subspectrum_filename), {'spectrum': transformed_spectrum_before})
    subspectrum_filename_1 = f"resonance_{filename}_spectrum_full_region.mat"
    savemat(os.path.join(save_path, subspectrum_filename_1), {'spectrum': transformed_spectrum_after})
    subspectrum_filename_2 = f"resonance_{filename}_spectrum_apply_mask1.mat"
    savemat(os.path.join(save_path, subspectrum_filename_2), {'spectrum': transformed_spectrum_after_roi})
    subspectrum_filename_3 = f"resonance_{filename}_spectrum_apply_mask0.mat"
    savemat(os.path.join(save_path, subspectrum_filename_3), {'spectrum': transformed_spectrum_after_roi_1})

    print(f'Subspectrum saved!')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6,8))
    ax1.plot(wavelengths, transformed_spectrum_before, label='Before_Spectrum', color='g')
    ax1.set_title('Before_Spectrum')
    # ax1.set_xlabel('Wavenumber (cm⁻¹)')
    # ax1.set_ylabel('Intensity')
    # ax1.legend(loc='upper left')

    ax2.plot(wavelengths, transformed_spectrum_after, label='Full_region_Spectrum', color='r')
    ax2.set_title('Full_region_Spectrum')
    # ax2.legend(loc='upper left')

    ax3.plot(wavelengths, transformed_spectrum_after_roi, label='Mask1_region_Spectrum', color='b')
    ax3.set_title('Mask1_region_Spectrum')
    # ax3.legend(loc='upper left')

    ax4.plot(wavelengths, transformed_spectrum_after_roi_1, label='Mask0_region_Spectrum', color='k')
    ax4.set_title('Mask0_region_Spectrum')
    # ax4.legend(loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(save_path, f'{filename}_spectrums.png'))
    # plt.show()
    plt.close()

if __name__ == '__main__':
    main()
