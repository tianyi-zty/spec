__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def main():

    spectrum_path = '../res/AuPillars_Al2O3_12102014/ALS/1/ALS_corrected_flipped_spectrum_95-5.mat'
    save_path = '../res/AuPillars_Al2O3_12102014/ALS/1'
    # Wavelength range (cm⁻¹)
    wavelength_start = 950
    wavelength_end = 1800
    wavelength_start_plot = 1300
    wavelength_end_plot = 1600
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat file
    data = loadmat(spectrum_path)

    spectra = data['corrected_spectrum'].flatten()

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    z_indices_plot = np.where((wavelengths >= wavelength_start_plot) & (wavelengths <= wavelength_end_plot))[0]
    
    # # Apply Savitzky-Golay filter to calculate the second-order derivative directly
    second_derivative = savgol_filter(spectra, window_length=13, polyorder=2, deriv=2) 
    # second_derivative_ref = savgol_filter(final_flipped_ref, window_length=13, polyorder=2, deriv=2)
    # Identify local minima by finding peaks in the inverted second derivative
    minima_indices, _ = find_peaks(-second_derivative, prominence=0.05)
    minima_indices = minima_indices[(minima_indices >= 0) & (minima_indices <= 426)]
    # st()
    minima_x = wavelengths[z_indices][minima_indices]
    minima_y = second_derivative[minima_indices]
    # minima_indices_ref, _ = find_peaks(-second_derivative_ref, prominence=0.0005)
    # minima_indices_ref = minima_indices_ref[(minima_indices_ref >= 236) & (minima_indices_ref <= 367)]
    # minima_x_ref = wavelengths[z_indices][minima_indices_ref]
    # minima_y_ref = second_derivative_ref[minima_indices_ref]
    # st()
    plt.figure(figsize=(20, 14))
    # plt.plot(wavelengths[z_indices], final_flipped, label='Final Flipped A', color='k', linewidth=2)
    plt.plot(wavelengths[z_indices_plot], second_derivative[z_indices_plot], label='Second Derivative of the spectrum', color='k', linewidth=2)
    plt.scatter(minima_x, minima_y, color='red', marker='o', label='Local Minima')
    # plt.plot(wavelengths[z_indices_defined], second_derivative_ref[z_indices_defined], label='Second Derivative of collagen (ref)', color='b', linewidth=2)
    # plt.scatter(minima_x_ref, minima_y_ref, color='green', marker='o', label='Local Minima')

    # Annotate each minimum with its (x, y) coordinate
    for x, y in zip(minima_x, minima_y):
        plt.annotate(f'({x:.0f}, {y:.4f})', 
                    xy=(x, y), 
                    xytext=(x, y),  # Adjust the y-coordinate for better visibility
                    fontsize=12, 
                    ha='center', 
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        
    # for x, y in zip(minima_x_ref, minima_y_ref):
    #     plt.annotate(f'({x:.2f}, {y:.5f})', 
    #                 xy=(x, y), 
    #                 xytext=(x, y),  # Adjust the y-coordinate for better visibility
    #                 fontsize=8, 
    #                 ha='center', 
    #                 arrowprops=dict(arrowstyle='->', color='green', lw=0.5))
    # st()
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc="upper right")
    # plt.show()
    # st()
    plt.savefig(os.path.join(save_path, 'Second_Derivative_with_coordinates_95-5.png'))
    # st()



if __name__ == '__main__':
    main()