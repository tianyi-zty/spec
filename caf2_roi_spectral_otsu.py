import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
from skimage.filters import threshold_multiotsu

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

############################## change here manully shift#####################################
def process_block(block_after, wavelengths, save_path, block_id):
    """
    Process a 10x10 block of spectra, save the results.
    """
    # Calculate the mean spectrum for the block
    mean_spectrum_after = np.mean(block_after, axis=(0, 1))
    ##T=10^-A
    # transformed_spectrum_after = 10 ** (-mean_spectrum_after)
    # st()
    # # Subtract shifted spectra ############################## change here wavelength range#####################################
    # wavelength_start_index = 0
    # wavelength_end_index = 426
    # subspectrum = transformed_spectrum_after[wavelength_start_index:wavelength_end_index]

    # Save the subspectrum
    subspectrum_filename = f"block_{block_id}.mat"
    savemat(os.path.join(save_path+'/subspectrum', subspectrum_filename), {'spectrum': mean_spectrum_after})
    print(f'Subspectrum for block {block_id} saved!')


    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, mean_spectrum_after, label='Spectrum (caf2)', color='r')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc='upper left')
    plt.title(f"Spectrum for Block {block_id}")
    plt.savefig(os.path.join(save_path+'/figures/', f"Block_{block_id}_Spectrum.png"))
    plt.close()

def main():


    filename = 'LMT_4'
    # before_collagen = r'/Volumes/TIANYI/Sperodata/CaF2_01162025/after_sample/LMT_1.mat'
    after_collagen = r'W:/3. Students/Tianyi/caf2_06132025/1000/rinse/bgcorrect/'+f'{filename}'+'.mat'
    save_path = f'../res/caf2_06132025/bgcorrect/1000/{filename}'
    os.makedirs(save_path, exist_ok=True)


    wavelengths = np.linspace(950, 1800, 426)
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))

    x_start, x_end = 0,480
    y_start, y_end = 0,480
    region_after = spectra_after[x_start:x_end, y_start:y_end, :]

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax1.imshow(spectra_after[:,:,330].T)
    ax1.set_title('spectra')
    ax2.imshow(region_after[:,:,330].T)
    ax2.set_title('spectra')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'spectra image visualization.png'))
    plt.show()
    # st()

    mean_spectrum = np.mean(region_after, axis=(0, 1))
    std_spectrum = np.std(region_after, axis=(0, 1))
    # block_size = 100
    # block_id = 0

    # for i in range(0, region_after.shape[0], block_size):
    #     for j in range(0, region_after.shape[1], block_size):
    #         block_after = region_after[i:i+block_size, j:j+block_size, :]
    #         # st()
    #         if block_after.shape[0] == block_size and block_after.shape[1] == block_size:
    #             process_block(block_after, wavelengths, save_path, block_id)
    #             block_id += 1
    #         st()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))  # Create a 2x2 grid of subplots
    ax1.imshow(region_after[:,:,330].T)
    ax1.set_title('spectra')
    # Step 1: Compute Multi-Otsu thresholds
    # Region 0: pixels < thresh1
    # Region 1: thresh1 <= pixels < thresh2
    # Region 2: pixels >= thresh2
    thresholds = threshold_multiotsu(region_after[:, :, 330].T, classes=3)
    print(thresholds) 
    # thresholds = np.array([0.05, 0.1, 0.8])
    regions = np.digitize(region_after[:, :, 330].T, bins=thresholds)
    # st()
    binary_mask_region0 = (regions == 0).astype(np.uint8)
    binary_mask_region1 = (regions == 1).astype(np.uint8)
    binary_mask_region2 = (regions == 2).astype(np.uint8)
    
    ax2.imshow(binary_mask_region1, cmap='gray')
    ax2.set_title('Binary Mask - Region 1')
    ax3.imshow(binary_mask_region2, cmap='gray')
    ax3.set_title('Binary Mask - Region 2')
    ax4.imshow(binary_mask_region0, cmap='gray')
    ax4.set_title('Binary Mask - Region 0')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{filename}_spatial_image.png'))
    plt.show()
    st()
    
    mean_spectrum_after = np.mean(binary_mask_region0.T[:, :, np.newaxis]*region_after, axis=(0, 1))
    std_spectrum_after = np.std(binary_mask_region0.T[:, :, np.newaxis]*region_after, axis=(0, 1))
    mean_spectrum_after_roi = np.mean(binary_mask_region1.T[:, :, np.newaxis]*region_after, axis=(0, 1))
    std_spectrum_after_roi = np.std(binary_mask_region1.T[:, :, np.newaxis]*region_after, axis=(0, 1))
    mean_spectrum_after_roi_1 = np.mean(binary_mask_region2.T[:, :, np.newaxis]*region_after, axis=(0, 1))
    std_spectrum_after_roi_1 = np.std(binary_mask_region2.T[:, :, np.newaxis]*region_after, axis=(0, 1))

    # Plot the average spectrum with standard deviation
    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, mean_spectrum_after, label='Average Spectrum', color='b', linewidth=2) 
    plt.fill_between(wavelengths, 
                    mean_spectrum_after - std_spectrum_after, 
                    mean_spectrum_after + std_spectrum_after, 
                    color='b', alpha=0.3, label='Standard Deviation')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'mask0_average_spectrum_{filename}.png'))

    # Plot the average spectrum with standard deviation
    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, mean_spectrum_after_roi, label='Average Spectrum', color='b', linewidth=2) 
    plt.fill_between(wavelengths, 
                    mean_spectrum_after_roi - std_spectrum_after_roi, 
                    mean_spectrum_after_roi + std_spectrum_after_roi, 
                    color='b', alpha=0.3, label='Standard Deviation')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'mask1_average_spectrum_{filename}.png'))

    # Plot the average spectrum with standard deviation
    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, mean_spectrum_after_roi_1, label='Average Spectrum', color='b', linewidth=2) 
    plt.fill_between(wavelengths, 
                    mean_spectrum_after_roi_1 - std_spectrum_after_roi_1, 
                    mean_spectrum_after_roi_1 + std_spectrum_after_roi_1, 
                    color='b', alpha=0.3, label='Standard Deviation')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'mask2_average_spectrum_{filename}.png'))

    # Save the subspectrum
    subspectrum_filename = f"resonance_{filename}.mat"
    savemat(os.path.join(save_path, subspectrum_filename), {'spectrum': mean_spectrum})
    subspectrum_filename_1 = f"resonance_{filename}_after_mask0.mat"
    savemat(os.path.join(save_path, subspectrum_filename_1), {'spectrum': mean_spectrum_after})
    subspectrum_filename_2 = f"resonance_{filename}_after_mask1.mat"
    savemat(os.path.join(save_path, subspectrum_filename_2), {'spectrum': mean_spectrum_after_roi})
    subspectrum_filename_3 = f"resonance_{filename}_after_mask2.mat"
    savemat(os.path.join(save_path, subspectrum_filename_3), {'spectrum': mean_spectrum_after_roi_1})

    print(f'Subspectrum saved!')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6,8))
    ax1.plot(wavelengths, mean_spectrum, label='before', color='g')
    ax1.set_title('before')
    ax2.plot(wavelengths, mean_spectrum_after, label='after_mask0', color='r')
    ax2.set_title('after_mask0')
    ax3.plot(wavelengths, mean_spectrum_after_roi, label='after_mask1', color='b')
    ax3.set_title('after_mask1')
    ax4.plot(wavelengths, mean_spectrum_after_roi_1, label='after_mask2', color='k')
    ax4.set_title('after_mask2')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(save_path, f'{filename}_spectrums.png'))
    # plt.show()
    plt.close()

if __name__ == '__main__':
    main()
