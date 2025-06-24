import os
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
from skimage.filters import threshold_multiotsu

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


    filename = 'LMT_1'
    # before_collagen = r'/Volumes/TIANYI/Sperodata/CaF2_01162025/after_sample/LMT_1.mat'
    after_collagen = r'W:/3. Students/Tianyi/Caf2_06232025/9010/'+f'{filename}'+'.mat'
    save_path = f'../res/Caf2_06232025_tnse/original/9010/{filename}/'
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
    data = binary_mask_region2.T[:, :, np.newaxis]*region_after


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))  # Create a 2x2 grid of subplots
    ax1.imshow(region_after[:,:,330].T)
    ax1.set_title('spectra')
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


    # Initialize counter
    count = 0

    # Loop through each pixel
    for i in range(data.shape[0]):       # height
        for j in range(data.shape[1]):   # width
            spectrum = data[i, j, :]     # shape (426,)
            if np.any(spectrum):         # only save if not all zeros
                np.save(os.path.join(save_path, f"spectrum_{count}.npy"), spectrum)
                # # Process the spectrum
                # second_derivative, minima_x, minima_y = process_spectrum(
                #     spectrum, wavelengths, wavelength_start, wavelength_end
                # )
                # np.save(os.path.join(save_2nd, f"2ndspectrum_{count}.npy"), second_derivative)
                count += 1

    print(f"Saved {count} non-zero spectra.")



if __name__ == '__main__':
    main()
