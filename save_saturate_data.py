import os
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter, find_peaks
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_multiotsu
import random


def save_spectrum_to_mat(spectrum, filename, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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
    foldername_list = ['1000'] #'1000', '9010', 
    filename_list = ['LMT_SATURATED']

    for foldername in foldername_list:
        for filename in filename_list:
            print('Processing:', foldername, filename)

            after_collagen = f'W:/3. Students/Tianyi/Caf2_07032025/second test/{foldername}/{filename}.mat'
            save_path = f'../res/Caf2_07032025_second_test_saturate/{foldername}/{filename}/'
            os.makedirs(save_path, exist_ok=True)

            data_after = loadmat(after_collagen)
            spectra_after = np.reshape(data_after['r'], (480, 480, 426))
            wavelengths = np.linspace(950, 1800, 426)
            wavelengths_270 = wavelengths[:270]

            region_after = spectra_after[0:480, 0:480, :]

            # Multi-Otsu thresholding on the 330th band
            thresholds = threshold_multiotsu(region_after[:, :, 330].T, classes=3)
            regions = np.digitize(region_after[:, :, 330].T, bins=thresholds)

            # Binary masks
            binary_mask_region0 = (regions == 0).astype(np.uint8)
            binary_mask_region1 = (regions == 1).astype(np.uint8)
            binary_mask_region2 = (regions == 2).astype(np.uint8)

            # Apply mask (Region 2 only)
            data = binary_mask_region2.T[:, :, np.newaxis] * region_after
            # combined_mask = (binary_mask_region1 + binary_mask_region2).T[:, :, np.newaxis]
            # data = combined_mask * region_after

            # Visualize masks
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
            ax1.imshow(region_after[:, :, 330].T)
            ax1.set_title('Spectra (Band 330)')
            ax2.imshow(binary_mask_region1, cmap='gray')
            ax2.set_title('Binary Mask - Region 1')
            ax3.imshow(binary_mask_region2, cmap='gray')
            ax3.set_title('Binary Mask - Region 2')
            ax4.imshow(binary_mask_region0, cmap='gray')
            ax4.set_title('Binary Mask - Region 0')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{filename}_spatial_image.png'))
            plt.close()

            # Step 1: Reshape to (num_pixels, 426), truncate to 270 bands
            flat_data = data.reshape(-1, data.shape[-1])[:, :270]
            non_zero_mask = np.any(flat_data, axis=1)
            valid_spectra = flat_data[non_zero_mask]

            # Step 2: Filter by Amide III band max intensity (1300–1350 cm⁻¹)
            amide3_mask = (wavelengths_270 >= 1300) & (wavelengths_270 <= 1350)
            filtered_spectra = []

            for spectrum in valid_spectra:
                max_amide3 = np.max(spectrum[amide3_mask])
                if 0.8 <= max_amide3 <= 2:
                    filtered_spectra.append(spectrum)

            print(f"Found {len(filtered_spectra)} spectra matching Amide III band intensity criteria.")

            # Step 3: Randomly save up to 10,000 filtered spectra
            n_samples = min(len(filtered_spectra), 8000)
            random.seed(42)
            selected_spectra = random.sample(filtered_spectra, n_samples)

            for count, spectrum in enumerate(selected_spectra):
                np.save(os.path.join(save_path, f"spectrum_{count}.npy"), spectrum)

            print(f"Randomly saved {n_samples} filtered spectra.")


if __name__ == "__main__":
    main()
