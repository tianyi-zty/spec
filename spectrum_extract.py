__author__ = 'Tianyi'

import spectrochempy as scp
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import os
import numpy as np


def main():

    before_collagen = r'W:/3. Students/TianYi/Caf2_11082024/beforecollagen/LMT_1.mat'
    after_collagen = r'W:/3. Students/TianYi/Caf2_11152024/aftercollagen/LMT_1.mat'
    save_path = '../res/Caf2_11152024/'
    name = 'LMT_1'
    # st()
    # Wavelength range (cm⁻¹)
    wavelength_start = 950
    wavelength_end = 1800
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat file
    data_before = loadmat(before_collagen)
    data_after = loadmat(after_collagen)
    spectra_before = np.reshape(data_before['r'], (480, 480, 426))
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))

    # Define the region of interest on the x and y axes
    x_start, x_end = 100, 400 #30, 230 #250, 350 # #  # Replace with your desired x range
    y_start, y_end = 100, 400  # Replace with your desired y range

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]

    # Extract the subregion and wavelength range
    extracted_region_before = spectra_before[x_start:x_end, y_start:y_end, z_indices]
    extracted_region_after = spectra_after[x_start:x_end, y_start:y_end, z_indices]

    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_before = np.mean(extracted_region_before, axis=(0, 1))
    std_spectrum_before = np.std(extracted_region_before, axis=(0, 1))
    mean_spectrum_after = np.mean(extracted_region_after, axis=(0, 1))
    std_spectrum_after = np.std(extracted_region_after, axis=(0, 1))
    # st()
    # Plot the average spectrum with standard deviation
    plt.figure(figsize=(12, 8))
    
    # # Plot the mean spectrum as a solid line
    # plt.plot(wavelengths[z_indices], mean_spectrum_before, label='Average Spectrum before', color='b', linewidth=2)

    # # Fill the area representing standard deviation
    # plt.fill_between(wavelengths[z_indices], 
    #                 mean_spectrum_before - std_spectrum_before, 
    #                 mean_spectrum_before + std_spectrum_before, 
    #                 color='b', alpha=0.3, label='Standard Deviation before')
    
    plt.plot(wavelengths[z_indices], mean_spectrum_after, label='Average Spectrum after', color='r', linewidth=2)
    # Fill the area representing standard deviation
    plt.fill_between(wavelengths[z_indices], 
                    mean_spectrum_after - std_spectrum_after, 
                    mean_spectrum_after + std_spectrum_after, 
                    color='r', alpha=0.3, label='Standard Deviation after')

    # Labeling the plot
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Average Spectrum and Standard Deviation in Region ({x_start}:{x_end}, {y_start}:{y_end})')
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.show()
    # Save
    plt.savefig(os.path.join(save_path, f'average_spectrum_region_{name}_{x_start}_{x_end}_{y_start}_{y_end}.png'))
    print(f"Spectrum saved.")

if __name__ == '__main__':
    main()