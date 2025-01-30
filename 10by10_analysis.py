import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st

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
def process_block(block_before, block_after, wavelengths, save_path, block_id, wv1, shift_value=0):
    """
    Process a 10x10 block of spectra, calculate the shift, and save the results.
    """
    # Calculate the mean spectrum for the block
    mean_spectrum_before = np.mean(block_before, axis=(0, 1))
    mean_spectrum_after = np.mean(block_after, axis=(0, 1))
    
    ##T=10^-A
    transformed_spectrum_before = 10 ** (-mean_spectrum_before)+0.04
    transformed_spectrum_after = 10 ** (-mean_spectrum_after)

    # Cross-correlation
    correlation = correlate(transformed_spectrum_before, transformed_spectrum_after, mode='full')
    lag = np.argmax(correlation) - (len(transformed_spectrum_after) - 1)
    x_step = wavelengths[1] - wavelengths[0]
    x_shift = lag * x_step
    print(f"Block {block_id}: Shift = {x_shift:.0f}+{shift_value} units")

    # Subtract shifted spectra ############################## change here wavelength range#####################################
    wavelength_start_index = int((wv1 - 1 - 950) / 2)
    wavelength_end_index = int((wv1+200 - 950) / 2)
    # subspectrum = transformed_spectrum_before[wavelength_start_index:wavelength_end_index] - \
    subspectrum =   transformed_spectrum_after[wavelength_start_index-int((x_shift + shift_value)/2):
                                              wavelength_end_index-int((x_shift + shift_value)/2)]

    # Save the subspectrum
    subspectrum_filename = f"block_{block_id}.mat"
    savemat(os.path.join(save_path+'/subspectrum', subspectrum_filename), {'spectrum': subspectrum})
    print(f'Subspectrum for block {block_id} saved!')

    # Save plots
    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, transformed_spectrum_before, label='Before Collagen', color='b')
    plt.plot(wavelengths, transformed_spectrum_after, label='After Collagen (flipped)', color='g')
    plt.plot(wavelengths + x_shift + shift_value, transformed_spectrum_after, label='Shifted Spectrum', linestyle='--', color='g')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc='upper left')
    plt.title(f"Spectra Shift {x_shift:.0f}+{shift_value} for Block {block_id}")
    plt.savefig(os.path.join(save_path+'/figures/', f"Block_{block_id}_Shifted.png"))
    plt.close()

    # plt.figure(figsize=(12, 8))
    # plt.plot(wavelengths[wavelength_start_index:wavelength_end_index], subspectrum, label='ROI Spectrum', color='r')
    # plt.xlabel('Wavenumber (cm⁻¹)')
    # plt.ylabel('Intensity')
    # plt.legend(loc='upper left')
    # plt.title(f"ROI Spectrum (Before - After) for Block {block_id}")
    # plt.savefig(os.path.join(save_path+'/figures/', f"Block_{block_id}_ROI_Spectrum.png"))
    # plt.close()
    # st()

def main():
    wv1=1600
    filename = f'{wv1}-{wv1+200}'
    before_collagen = r'/Volumes/TIANYI/Sperodata/AuPillars_10nmAl2O3_01162025/before_sample/LMR_2.mat'
    after_collagen = r'/Volumes/TIANYI/Sperodata/AuPillars_10nmAl2O3_01162025/after_sample/LMR_2.mat'
    save_path = f'../res/AuPillars_10nmAl2O3_01162025/2ndafter/{filename}'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'/subspectrum', exist_ok=True)
    os.makedirs(save_path+'/figures', exist_ok=True)

    wavelengths = np.linspace(950, 1800, 426)
    data_before = loadmat(before_collagen)
    spectra_before = np.reshape(data_before['r'], (480, 480, 426))
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))

    x_start, x_end = 100, 220 #280, 400 #
    y_start, y_end = 140, 260 #150, 270 #
    x_start_1, x_end_1 = 280, 400 #100, 220 #
    y_start_1, y_end_1 = 150, 270 #140, 260 #
    region_before = spectra_before[x_start_1:x_end_1, y_start_1:y_end_1, :]
    region_after = spectra_after[x_start:x_end, y_start:y_end, :]


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
    (ax1, ax2), (ax3, ax4) = axes  # Unpack axes for easier reference
    ax1.imshow(spectra_before[:,:,330])
    ax1.set_title('spectra_before')
    ax2.imshow(spectra_after[:,:,330])
    ax2.set_title('spectra_after')
    ax3.imshow(region_before[:,:,330])
    ax3.set_title('extracted_region_before')
    ax4.imshow(region_after[:,:,330])
    ax4.set_title('extracted_region_after')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_path, 'cropped image example.png'))
    st()
    block_size = 20
    block_id = 0

    for i in range(0, region_before.shape[0], block_size):
        for j in range(0, region_before.shape[1], block_size):
            block_before = region_before[i:i+block_size, j:j+block_size, :]
            block_after = region_after[i:i+block_size, j:j+block_size, :]
            # st()
            if block_before.shape[0] == block_size and block_before.shape[1] == block_size:
                process_block(block_before, block_after, wavelengths, save_path, block_id, wv1)
                block_id += 1

if __name__ == '__main__':
    main()
