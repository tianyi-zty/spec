__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import os
import numpy as np


def main():

    before_collagen = r'../res/after_cleaning/AuPilllars_10nmAl2O3_Cleaning_05122025/1-100/6/spectral_data_1.mat' # 100:0 collagen 1:peptide
    after_collagen = r'../res/after_cleaning/AuPillars_50nmAl2O3_2_05222025/6/spectral_data_1.mat' # 80:20 collagen 1:peptide
    after_collagen_1 = r'../res/after_cleaning/AuPillars_50nmAl2O3_6_05232025/6/spectral_data_1.mat' # 50:50 collagen 1:peptide
    save_path = '../res/after_cleaning/AuPillars_collagen1andpeptide1'
    os.makedirs(save_path, exist_ok=True)
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat file
    data_before = loadmat(before_collagen)
    data_after = loadmat(after_collagen)  
    data_after_1 = loadmat(after_collagen_1)  
    # st()
    spectra_before = np.reshape(data_before['corrected_spectrum'], (426))
    spectra_after = np.reshape(data_after['corrected_spectrum'], (426))
    spectra_after_1 = np.reshape(data_after_1['corrected_spectrum'], (426))

    # Define the region of interest on the x and y axes
    x_start, x_end = 1200, 1400 #30, 230 #250, 350 # #  # Replace with your desired x range
    # y_start, y_end = 100, 400  # Replace with your desired y range

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= x_start) & (wavelengths <= x_end))[0]

    # Extract the subregion and wavelength range
    extracted_region_before = spectra_before[z_indices]
    extracted_region_after = spectra_after[z_indices]
    extracted_region_after_1 = spectra_after_1[z_indices]
    # st()
    sv=0
    # Plot the average spectrum with standard deviation
    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths[z_indices], extracted_region_before, label='Spectrum collagen1:peptide 100:0', color='r', linewidth=2)
    plt.plot(wavelengths[z_indices]-sv, extracted_region_after, label='Spectrum  80:20', color='b', linewidth=2)
    plt.plot(wavelengths[z_indices]-sv, extracted_region_after_1, label='Spectrum 50:50', color='g', linewidth=2)
    # Labeling the plot
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    # plt.title(f'Spectrum in Region ({x_start}-{x_end});Spectrum 95:5 shift left({sv})')
    plt.legend(loc="upper right")
    plt.grid(True)
    # 
    # Save
    plt.savefig(os.path.join(save_path, f'Spectrum in Region ({x_start}-{x_end}).png'))
    plt.show()
    print(f"Spectrum saved.")

if __name__ == '__main__':
    main()