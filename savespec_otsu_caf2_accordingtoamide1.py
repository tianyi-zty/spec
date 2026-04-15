import os
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
from skimage.filters import threshold_multiotsu
import random


#######save single pixel spectrum (after bg corection, baseline correction, otsu mask) for tsne######
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


def process_spectrum(spectrum, wavelengths, start, end, window=13, polyorder=2, prominence=0.0003):
        indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
        second_derivative = savgol_filter(spectrum, window_length=window, polyorder=polyorder, deriv=2)
        minima_indices, _ = find_peaks(-second_derivative, prominence=prominence)
        minima_x = wavelengths[indices][minima_indices]
        minima_y = second_derivative[minima_indices]
        return second_derivative, minima_x, minima_y

def main():
    foldername_list = ['9010SER','9010PSER','8020PSER','7030PSER','6040PSER','1000'] # ['9010SER','9010PSER','8020SER','8020PSER','7030SER','7030PSER','6040SER','6040PSER','1000'] # '1000','9010', '8020','7030' 
    filename_list = ['LMT_1','LMT_2'] #'LMT_1','LMT_2','LMT_3','LMT_4'
    for foldername in foldername_list:
        for filename in filename_list:
            print('processing:', foldername, filename)
            # before_collagen = r'/Volumes/TIANYI/Sperodata/CaF2_01162025/after_sample/LMT_1.mat'
            after_collagen = f'W:/3. Students/Tianyi/Caf2_11102025/{foldername}/{filename}'+'.mat'
            save_path = f'../res/Caf2_11102025/{foldername}/{filename}/'
            os.makedirs(save_path, exist_ok=True)
            save_2nd = f'../res/Caf2_11102025_2nd/{foldername}/{filename}/'
            os.makedirs(save_2nd, exist_ok=True)

            data_after = loadmat(after_collagen)
            spectra_after = np.reshape(data_after['r'], (480, 480, 426))
            wavelengths = np.linspace(950, 1800, 426)  # Assuming this range for all subspectra
            wavelength_start = 950
            wavelength_end = 1800

            x_start, x_end = 0,480
            y_start, y_end = 0,480
            region_after = spectra_after[x_start:x_end, y_start:y_end, :]

            # Step 1: Compute Multi-Otsu thresholds
            # Region 0: pixels < thresh1
            # Region 1: thresh1 <= pixels < thresh2
            # Region 2: pixels >= thresh2
            thresholds = threshold_multiotsu(region_after[:, :, 67].T, classes=4)
            # print(thresholds) 
            # thresholds = np.array([0.05, 0.1, 0.8])
            regions = np.digitize(region_after[:, :, 330].T, bins=thresholds)
            # st()
            binary_mask_region0 = (regions == 0).astype(np.uint8)
            binary_mask_region1 = (regions == 1).astype(np.uint8)
            binary_mask_region2 = (regions == 2).astype(np.uint8)
            binary_mask_region3 = (regions == 3).astype(np.uint8)
            # binary_mask_region4 = (regions == 4).astype(np.uint8)
            # data = binary_mask_region0.T[:, :, np.newaxis]*region_after  ### + binary_mask_region2
            combined_mask = (binary_mask_region2+ binary_mask_region3).T[:, :, np.newaxis]
            data = combined_mask * region_after

            fig, ((ax1, ax2, ax3 ), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9, 6))  # Create a 2x2 grid of subplots
            ax1.imshow(region_after[:,:,330].T)
            ax1.set_title('spectra')
            ax2.imshow(binary_mask_region0, cmap='gray')
            ax2.set_title('Binary Mask - Region 0')
            ax3.imshow(binary_mask_region1, cmap='gray')
            ax3.set_title('Binary Mask - Region 1')
            ax4.imshow(binary_mask_region2, cmap='gray')
            ax4.set_title('Binary Mask - Region 2')
            # ax5 = fig.add_subplot(2, 3, 5)
            ax5.imshow(binary_mask_region3, cmap='gray')
            ax5.set_title('Binary Mask - Region 3')
            # ax5 = fig.add_subplot(2, 3, 2)
            # ax5.imshow(binary_mask_region4, cmap='gray')
            # ax5.set_title('Binary Mask - Region 4')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{filename}_spatial_image.png'))
            plt.show()
            st()
            print ("Region 1 2 have been saved, at 1240cm-1 mask.")

            ############## Extract the Amide I region (1600–1700 cm⁻¹),Check if the maximum intensity is between 0.8 and 0.95 max amide, and Save only those spectra.
            # Define Amide I band range
            amide1_start = 1600
            amide1_end = 1700

            # Find indices corresponding to Amide I range
            amide1_indices = np.where((wavelengths >= amide1_start) & (wavelengths <= amide1_end))[0]

           
            # st()
            # #  first get the max amide 1 intensity
            # amide_spectra = []
            # for i in range(data.shape[0]):
            #     for j in range(data.shape[1]):
            #         spectrum = data[i, j, :]
            #         if not np.any(spectrum):
            #             continue
            #         amide1_band = spectrum[amide1_indices]
            #         # max_amide1 = np.max(amide1_band)
            #         # print(max_amide1)
            #         amide_spectra.append(amide1_band)
            # max_amide1 = np.max(amide_spectra)

            # Collect spectra satisfying Amide I condition: 0.8-0.95 max
            valid_spectra = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    spectrum = data[i, j, :]
                    if not np.any(spectrum):
                        continue
                    amide1_band = spectrum[amide1_indices]
                    # st()
                    print(np.max(amide1_band))
                    if 0.6 <= np.max(amide1_band)<=0.8:
                        valid_spectra.append(spectrum)

            print(f"Found {len(valid_spectra)} spectra with Amide I max between xxx in {foldername}/{filename}")

            # Randomly sample up to 10,000
            random.seed(42)
            n_samples = min(len(valid_spectra), 2000)
            selected_spectra = random.sample(valid_spectra, n_samples)

            # Save selected spectra
            for count, spectrum in enumerate(selected_spectra):
                second_derivative, minima_x, minima_y = process_spectrum(spectrum, wavelengths, 950, 1800, window=13, polyorder=2, prominence=0.0003)
                np.save(os.path.join(save_2nd, f"2nd_{count}.npy"), second_derivative)
                np.save(os.path.join(save_path, f"spectrum_{count}.npy"), spectrum)

            print(f"✅ Randomly saved {n_samples} spectra in {save_path} and {save_2nd}.")

            


if __name__ == '__main__':
    main()
