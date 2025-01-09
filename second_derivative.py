__author__ = 'Tianyi'

import spectrochempy as scp
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks, savgol_filter

def main():

    spectrum_path = '../res/AuPillars_Al2O3_12102024/ALS/3/new_way/99-1/99-1ROI Spectrum1199-1600.mat'
    spectrum_path_1 = '../res/AuPillars_Al2O3_12102024/ALS/3/new_way/95-5/95-5ROI Spectrum1199-1600.mat'
    save_path = '../res/AuPillars_Al2O3_12102024/ALS/3/new_way/'
    # Wavelength range (cm⁻¹)
    wavelength_start = 1200
    wavelength_end = 1600
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat file
    data = loadmat(spectrum_path)
    spectra = data['corrected_spectrum'].flatten()
    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    # # Apply Savitzky-Golay filter to calculate the second-order derivative directly
    second_derivative = savgol_filter(spectra, window_length=13, polyorder=2, deriv=2) 
    # Identify local minima by finding peaks in the inverted second derivative
    minima_indices, _ = find_peaks(-second_derivative, prominence=0.001)
    # st()
    minima_x = wavelengths[z_indices][minima_indices]
    minima_y = second_derivative[minima_indices]
    print('detected peaks 99-1:', minima_x)
    # st()

    # Load the .mat file
    data_1 = loadmat(spectrum_path_1)
    spectra_1 = data_1['corrected_spectrum'].flatten()
    # Find the indices corresponding to the desired wavelength range
    z_indices_1 = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    # # Apply Savitzky-Golay filter to calculate the second-order derivative directly
    second_derivative_1 = savgol_filter(spectra_1, window_length=13, polyorder=2, deriv=2) 
    # Identify local minima by finding peaks in the inverted second derivative
    minima_indices_1, _ = find_peaks(-second_derivative_1, prominence=0.001)
    # st()
    minima_x_1 = wavelengths[z_indices_1][minima_indices_1]
    minima_y_1 = second_derivative_1[minima_indices_1]
    print('detected peaks 95-5:', minima_x_1)
    # st()

    plt.figure(figsize=(20, 14))
    # plt.plot(wavelengths[z_indices], final_flipped, label='Final Flipped A', color='k', linewidth=2)
    plt.plot(wavelengths[z_indices], second_derivative, label='Second Derivative of the spectrum 99-1', color='k', linestyle='--', linewidth=2)
    plt.scatter(minima_x, minima_y, color='red', marker='o', label='Local Minima')

    # Annotate each minimum with its (x, y) coordinate
    for x, y in zip(minima_x, minima_y):
        plt.annotate(f'({x:.0f}, {y:.4f})', 
                    xy=(x, y), 
                    xytext=(x, y),  # Adjust the y-coordinate for better visibility
                    fontsize=12, 
                    ha='center', 
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc="upper right")
    # plt.show()
    # st()
    # plt.savefig(os.path.join(save_path, 'Second_Derivative_with_coordinates_99-1.png'))
    # st()

    # plt.plot(wavelengths[z_indices], final_flipped, label='Final Flipped A', color='k', linewidth=2)
    plt.plot(wavelengths[z_indices_1], second_derivative_1, label='Second Derivative of the spectrum 95-5', color='k', linewidth=2)
    plt.scatter(minima_x_1, minima_y_1, color='red', marker='o', label='Local Minima')

    # Annotate each minimum with its (x, y) coordinate
    for x, y in zip(minima_x_1, minima_y_1):
        plt.annotate(f'({x:.0f}, {y:.4f})', 
                    xy=(x, y), 
                    xytext=(x, y),  # Adjust the y-coordinate for better visibility
                    fontsize=12, 
                    ha='center', 
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc="upper right")
    # plt.show()
    # st()
    plt.savefig(os.path.join(save_path, 'Second_Derivative_with_coordinates.png'))
    # st()



if __name__ == '__main__':
    main()