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


def process_spectrum(spectrum, wavelengths, start, end, window=13, polyorder=2, prominence=0.0002):
        indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
        second_derivative = savgol_filter(spectrum, window_length=window, polyorder=polyorder, deriv=2)
        minima_indices, _ = find_peaks(-second_derivative, prominence=prominence)
        minima_x = wavelengths[indices][minima_indices]
        minima_y = second_derivative[minima_indices]
        return second_derivative, minima_x, minima_y

def main():
    foldername_list = ['1000'] # '1000','9010', '8020','7030' 
    filename_list = ['LMT_1','LMT_2','LMT_3','LMT_4'] #'LMT_1','LMT_2','LMT_3','LMT_4']
    for foldername in foldername_list:
        for filename in filename_list:
            print('processing:', foldername, filename)
            # before_collagen = r'/Volumes/TIANYI/Sperodata/CaF2_01162025/after_sample/LMT_1.mat'
            after_collagen = f'W:/3. Students/Tianyi/Caf2_07032025/{foldername}/{filename}'+'.mat'
            save_path = f'../res/Caf2_07032025_amide1/{foldername}/{filename}/'
            # save_2nd = f'../res/Caf2_06232025_tnse/2nd/1000/{filename}/'
            os.makedirs(save_path, exist_ok=True)
            # os.makedirs(save_2nd, exist_ok=True)

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
            thresholds = threshold_multiotsu(region_after[:, :, 330].T, classes=3)
            # print(thresholds) 
            # thresholds = np.array([0.05, 0.1, 0.8])
            regions = np.digitize(region_after[:, :, 330].T, bins=thresholds)
            # st()
            binary_mask_region0 = (regions == 0).astype(np.uint8)
            binary_mask_region1 = (regions == 1).astype(np.uint8)
            binary_mask_region2 = (regions == 2).astype(np.uint8)
            data = binary_mask_region1.T[:, :, np.newaxis]*region_after
            # combined_mask = (binary_mask_region1 + binary_mask_region2).T[:, :, np.newaxis]
            # data = combined_mask * region_after

            # st()

            # # Create binary mask based on Amide I intensity
            # amide1_start = 1600
            # amide1_end = 1700
            # amide1_indices = np.where((wavelengths >= amide1_start) & (wavelengths <= amide1_end))[0]

            # # Initialize empty mask
            # amide1_mask = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)

            # # Loop over pixels to apply condition
            # for i in range(data.shape[0]):
            #     for j in range(data.shape[1]):
            #         spectrum = data[i, j, :]
            #         if not np.any(spectrum):
            #             continue
            #         max_amide1 = np.max(spectrum[amide1_indices])
            #         print(max_amide1)
            #         if 0.4 <= max_amide1 <= 1.2:
            #             amide1_mask[i, j] = 1

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))  # Create a 2x2 grid of subplots
            ax1.imshow(region_after[:,:,330].T)
            ax1.set_title('spectra')
            ax2.imshow(binary_mask_region1, cmap='gray')
            ax2.set_title('Binary Mask - Region 1')
            ax3.imshow(binary_mask_region2, cmap='gray')
            ax3.set_title('Binary Mask - Region 2')
            ax4.imshow(binary_mask_region0, cmap='gray')
            ax4.set_title('Binary Mask - Region 0')
            # ax5 = fig.add_subplot(2, 3, 5)
            # ax5.imshow(amide1_mask.T, cmap='gray')
            # ax5.set_title('Amide I Mask')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{filename}_spatial_image.png'))
            # plt.show()
            # st()


            ######### save all the mask 2 spectrum
            # count = 0

            # # Loop through each pixel
            # for i in range(data.shape[0]):       # height
            #     for j in range(data.shape[1]):   # width
            #         spectrum = data[i, j, :]     # shape (426,)
            #         if np.any(spectrum):         # only save if not all zeros
            #             np.save(os.path.join(save_path, f"spectrum_{count}.npy"), spectrum)
            #             # # Process the spectrum
            #             # second_derivative, minima_x, minima_y = process_spectrum(
            #             #     spectrum, wavelengths, wavelength_start, wavelength_end
            #             # )
            #             # np.save(os.path.join(save_2nd, f"2ndspectrum_{count}.npy"), second_derivative)
            #             count += 1

            # print(f"Saved {count} non-zero spectra.")

            ######## randomly save the mask 2 spectrum
            # # Collect all non-zero spectra
            # valid_spectra = []
            # random.seed(42)  # Ensures same results every run
            # for i in range(data.shape[0]):
            #     for j in range(data.shape[1]):
            #         spectrum = data[i, j, :]
            #         # st()
            #         if np.any(spectrum):
            #             valid_spectra.append(spectrum)

            # print(f"Found {len(valid_spectra)} valid spectra.")
            # # Randomly select up to 5000
            # n_samples = min(len(valid_spectra), 10000)
            # selected_spectra = random.sample(valid_spectra, n_samples)
            # # Save selected spectra
            # for count, spectrum in enumerate(selected_spectra):
            #     np.save(os.path.join(save_path, f"spectrum_{count}.npy"), spectrum)

            # print(f"Randomly saved {n_samples} spectra.")


            ############## Extract the Amide I region (1600–1700 cm⁻¹),Check if the maximum intensity is between 0.8 and 1.0, and Save only those spectra.
            # Define Amide I band range
            amide1_start = 1600
            amide1_end = 1700

            # Find indices corresponding to Amide I range
            amide1_indices = np.where((wavelengths >= amide1_start) & (wavelengths <= amide1_end))[0]

            # count = 0
            # for i in range(data.shape[0]):
            #     for j in range(data.shape[1]):
            #         spectrum = data[i, j, :]
            #         if not np.any(spectrum):
            #             continue  # Skip empty pixels

            #         # Extract Amide I region and check max intensity
            #         amide1_band = spectrum[amide1_indices]
            #         max_amide1 = np.max(amide1_band)

            #         if 0.9 <= max_amide1 <= 1.1:
            #             np.save(os.path.join(save_path, f"spectrum_{count}.npy"), spectrum)
            #             count += 1

            # print(f"Saved {count} spectra with max Amide I intensity between 0.9 and 1.1.")

            # Collect spectra satisfying Amide I condition
            valid_spectra = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    spectrum = data[i, j, :]
                    if not np.any(spectrum):
                        continue
                    amide1_band = spectrum[amide1_indices]
                    max_amide1 = np.max(amide1_band)
                    print(max_amide1)
                    if 0.5 <= max_amide1 <= 1.0:
                        valid_spectra.append(spectrum)

            print(f"Found {len(valid_spectra)} spectra with Amide I max between 0.6 and 1.2.")

            # Randomly sample up to 10,000
            random.seed(42)
            n_samples = min(len(valid_spectra), 5000)
            selected_spectra = random.sample(valid_spectra, n_samples)

            # Save selected spectra
            for count, spectrum in enumerate(selected_spectra):
                np.save(os.path.join(save_path, f"spectrum_{count}.npy"), spectrum)

            print(f"✅ Randomly saved {n_samples} spectra.")

            


if __name__ == '__main__':
    main()
