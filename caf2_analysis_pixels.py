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
<<<<<<< HEAD:caf2_analysis.py
    filename = 'HMT_2'
    # before_collagen = r'/Volumes/TIANYI/Sperodata/CaF2_01162025/after_sample/LMT_1.mat'
    after_collagen = r'/Volumes/TIANYI/Caf2_03132025_rat_ffpe/liver_ffpe/HMT_2.mat'
    save_path = f'../res/Caf2_03132025_rat_ffpe/liver_ffpe/{filename}'
=======


    filename = 'HMT_6'
    # before_collagen = r'/Volumes/TIANYI/Sperodata/CaF2_01162025/after_sample/LMT_1.mat'
    after_collagen = r'W:/3. Students/Tianyi/Caf2_03132025_rat_ffpe/liver_ffpe/'+f'{filename}'+'.mat'
    save_path = f'../res/Caf2_03072025_rat/liver_ffpe/{filename}'
    save_path_avg = f'../res/Caf2_03072025_rat/liver_ffpe/average'
    os.makedirs(save_path_avg, exist_ok=True)
>>>>>>> 9082f13a1562e3a793021b7d4696fa39f06dce59:caf2_analysis_pixels.py
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'/subspectrum', exist_ok=True)
    os.makedirs(save_path+'/figures', exist_ok=True)

    wavelengths = np.linspace(950, 1800, 426)
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))

    # x_start, x_end = 80, 200 
    # y_start, y_end = 100, 220 
    # region_after = spectra_after[x_start:x_end, y_start:y_end, :]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(spectra_after[:,:,330])
    ax.set_title('spectra')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_path, 'spectra image visualization.png'))
    # st()

    mean_spectrum = np.mean(spectra_after, axis=(0, 1))
    std_spectrum = np.std(spectra_after, axis=(0, 1))

    # Plot the average spectrum with standard deviation
    plt.figure(figsize=(12, 8))
    # Plot the mean spectrum as a solid line
    plt.plot(wavelengths, mean_spectrum, label='Average Spectrum', color='b', linewidth=2) 
    # Fill the area representing standard deviation
    plt.fill_between(wavelengths, 
                    mean_spectrum - std_spectrum, 
                    mean_spectrum + std_spectrum, 
                    color='b', alpha=0.3, label='Standard Deviation')
    # Labeling the plot
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Average Spectrum and Standard Deviation)')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # Save
    plt.savefig(os.path.join(save_path_avg, f'average_spectrum_{filename}.png'))
    print(f"Spectrum saved.")
    # st()


    block_size = 100
    block_id = 0

    for i in range(0, spectra_after.shape[0], block_size):
        for j in range(0, spectra_after.shape[1], block_size):
            block_after = spectra_after[i:i+block_size, j:j+block_size, :]
            # st()
            if block_after.shape[0] == block_size and block_after.shape[1] == block_size:
                process_block(block_after, wavelengths, save_path, block_id)
                block_id += 1
            # st()


if __name__ == '__main__':
    main()
