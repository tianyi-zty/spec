__author__ = 'Tianyi'
import os
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
import scipy.io as sio
import numpy as np
from scipy.io import savemat
from scipy.signal import correlate

######## this code we do not apply a ALS baseline correction #########

def save_spectrum_to_mat(spectrum, filename, save_path):
    """
    Save the spectrum to a .mat file at the specified path.
    
    Parameters:
        spectrum (array-like): The spectrum to save.
        filename (str): The name of the output .mat file.
        save_path (str): The directory path where the .mat file should be saved.
    """
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the directory if it doesn't exist
    
    # Create the full path to save the file
    full_path = os.path.join(save_path, filename)
    
    # Prepare the dictionary to save in .mat format
    data_dict = {'corrected_spectrum': spectrum}
    
    # Save the dictionary as a .mat file
    savemat(full_path, data_dict)
    print(f"Spectrum saved to {full_path}")

def main():

    filename = '95-5'
    before_collagen = r'W:/3. Students/Tianyi/AuPillars_10nmAl2O3_12102024/reference/LMR_1.mat'
    after_collagen = r'W:/3. Students/Tianyi/AuPillars_10nmAl2O3_12102024/'+f"{filename}"+'/after/LMR_1.mat'
    # collagen_ref = r'W:/3. Students/TianYi/AuPillars_11042024/afterCollagen/LMT_2.mat'
    save_path = '../res/AuPillars_Al2O3_12102024/ALS/1/pixel/'+f"{filename}"
    os.makedirs(save_path, exist_ok=True)
    # Create directories to save individual pixel spectra
    # os.makedirs(save_path+"/spectra_before", exist_ok=True)
    # os.makedirs(save_path+"/spectra_after", exist_ok=True)
    os.makedirs(save_path+"/subspectra", exist_ok=True)

    # Wavelength range (cm⁻¹)
    wavelength_start = 950
    wavelength_end = 1800
    wavelength_start_plot = 1400-1
    wavelength_end_plot = 1700
    wavelength_start_index = int((wavelength_start_plot-950)/2)
    wavelength_end_index = int((wavelength_end_plot-950)/2)
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat fileS
    data_before = loadmat(before_collagen)
    spectra_before = np.reshape(data_before['r'], (480, 480, 426))
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))
    
    # Define the region of interest on the x and y axes
    # x1_start, x1_end = 80, 200 #280, 400 # Replace with your desired y range
    # y1_start, y1_end = 80, 200 # Replace with your desired x range
    x1_start, x1_end = 110, 230 #280, 400 # # #270, 420   #250, 350 # 30, 230 # # Replace with your desired y range
    y1_start, y1_end = 90, 210 #150, 300  # Replace with your desired x range
    x2_start, x2_end = 300,420
    y2_start, y2_end = 90, 210

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    z_indices_plot = np.where((wavelengths >= wavelength_start_plot) & (wavelengths <= wavelength_end_plot))[0]
    # Extract the subregion and wavelength range
    extracted_region_before = spectra_before[x2_start:x2_end, y2_start:y2_end, z_indices]
    extracted_region_after = spectra_after[x1_start:x1_end, y1_start:y1_end, z_indices]
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_before = np.mean(extracted_region_before, axis=(0, 1))
    transformed_spectrum_before = 10 **(-mean_spectrum_before)
    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
    (ax1, ax2), (ax3, ax4) = axes  # Unpack axes for easier reference
    ax1.imshow(spectra_before[:,:,330])
    ax1.set_title('spectra_before')
    ax2.imshow(spectra_after[:,:,330])
    ax2.set_title('spectra_after')
    ax3.imshow(extracted_region_before[:,:,330])
    ax3.set_title('extracted_region_before')
    ax4.imshow(extracted_region_after[:,:,330])
    ax4.set_title('extracted_region_after')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_path, 'cropped image example.png'))

    
    shift_value = 16
    # Loop through each pixel in the region of interest
    for i in range(extracted_region_after.shape[0]):
        for j in range(extracted_region_after.shape[1]):
            # Extract the spectrum for the current pixel
            pixel_spectrum_after = extracted_region_after[i, j, :]

            transformed_spectrum_after = 10 **(-pixel_spectrum_after)

            # Cross-correlate the two spectra
            correlation = correlate(transformed_spectrum_before, transformed_spectrum_after, mode='full')
            lag = np.argmax(correlation) - (len(transformed_spectrum_after) - 1)

            # Convert lag to shift in terms of x-axis units (e.g., wavenumber, cm⁻¹)
            x_step = wavelengths[1] - wavelengths[0]  # Assumes equal spacing in x-axis values
            x_shift = lag * x_step
            print(f"The shift between the two spectra at pixel_{i}_{j} is approximately {x_shift:.2f} x-axis units.")
            # st()

            subspectrum = transformed_spectrum_before[wavelength_start_index: wavelength_end_index] - transformed_spectrum_after[wavelength_start_index-int((x_shift + shift_value)/2) : wavelength_end_index-int((x_shift + shift_value)/2)]

            # Save each pixel's spectrum to a .mat file
            # before_filename = f"/pixel_{i}_{j}.mat"
            # after_filename = f"/pixel_{i}_{j}.mat"
            subspectrum_filename = f"/pixel_{i}_{j}.mat"
            
            # sio.savemat(save_path+"/spectra_before/"+before_filename, {'spectrum': transformed_spectrum_before})
            # sio.savemat(save_path+"/spectra_after/"+after_filename, {'spectrum': transformed_spectrum_after})
            sio.savemat(save_path+"/subspectra/"+subspectrum_filename, {'spectrum': subspectrum})
            print(f'subspectrum at pixel_{i}_{j} saved!')
            # st()

            # #####uncomment to see the spectrum#####
            # plt.figure(figsize=(12, 8))
            # plt.plot(wavelengths[z_indices], transformed_spectrum_before, label='Baseline Corrected Spectrum - before collagen', color='b', linewidth=2)
            # plt.plot(wavelengths, transformed_spectrum_after, label='Flipped Spectrum', color='g', linewidth=2)
            # plt.plot(wavelengths + x_shift + shift_value, transformed_spectrum_after, label=f'Flipped Spectrum shift back {x_shift+shift_value} units', color='g', linestyle='--', linewidth=2)
            # plt.xlabel('Wavenumber (cm⁻¹)')
            # plt.ylabel('Intensity')
            # plt.legend(loc="upper left")
            # # plt.show()
            # # st()
            # plt.savefig(os.path.join(save_path, f"{filename}"+'Spectrum Shifted' f"{x_shift}" '+' f"{shift_value}"+ f'at pixel_{i}_{j}'+'.png'))

            # plt.figure(figsize=(12, 8))
            # plt.plot(wavelengths[z_indices_plot], subspectrum, label='ROI Spectrum', color='r')
            # plt.xlabel('Wavenumber (cm⁻¹)')
            # plt.ylabel('Intensity')
            # plt.legend(loc="upper left")
            # plt.title('ROI Spectrum (before - after)')
            # # plt.show()
            # st()
            # plt.savefig(os.path.join(save_path, f"{filename}"+'ROI Spectrum (before - after)'f"{x_shift}" '+' f"{shift_value}"+ f'at pixel_{i}_{j}'+'.png'))

if __name__ == '__main__':
    main()