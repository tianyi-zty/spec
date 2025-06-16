__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import correlate
from scipy.io import savemat


def rubberband_baseline_correction(x, y):
    """
    Perform rubberband baseline correction for a given spectrum.
    
    Parameters:
        x (array-like): The x-axis values (e.g., wavelength or wavenumber).
        y (array-like): The y-axis values (e.g., intensity).

    Returns:
        corrected_y (array-like): The baseline-corrected y values.
        baseline (array-like): The estimated baseline.
    """
    # Find the convex hull
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    
    # Extract the vertices of the convex hull
    v = hull.vertices
    
    # Rotate the convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())
    
    # Leave only the ascending part of the convex hull
    v = v[:v.argmax() + 1]
    
    # Create the baseline using linear interpolation between the vertices
    lower_baseline = np.interp(x, x[v], y[v])
    
    # Subtract the baseline from the original spectrum
    corrected_y = y - lower_baseline
    # st()
    return corrected_y, lower_baseline

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

    # before_collagen = r'W:/3. Students/Tianyi/AuPilllars_10nmAl2O3_Cleaning_05122025/before_2/LMR_1.mat'
    # after_collagen = r'W:/3. Students/Tianyi/AuPilllars_10nmAl2O3_Cleaning_05122025/after_2/LMR_2.mat'
    # save_path = '../res/after_cleaning/AuPilllars_10nmAl2O3_Cleaning_05122025/1-100/8'
    before_collagen = r'W:/3. Students/Tianyi/AuPillars_50nmAl2O3_6_05232025/before/LMR_2.mat'
    after_collagen = r'W:/3. Students/Tianyi/AuPillars_50nmAl2O3_6_05232025/after/LMR_3.mat'
    save_path = '../res/after_cleaning/AuPillars_50nmAl2O3_6_05232025/7'
    os.makedirs(save_path, exist_ok=True)

    # Wavelength range (cm⁻¹)
    wavelength_start = 950
    wavelength_end = 1800
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat file
    data_before = loadmat(before_collagen)
    spectra_before = np.reshape(data_before['r'], (480, 480, 426))
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))

    x_start, x_end = 150,250
    y_start, y_end = 350,450
    x_start_1, x_end_1 = 100,200
    y_start_1, y_end_1 = 350,450
    region_before = spectra_before[x_start:x_end, y_start:y_end, :]
    region_after = spectra_after[x_start_1:x_end_1, y_start_1:y_end_1, :]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
    (ax1, ax2), (ax3, ax4) = axes  # Unpack axes for easier reference
    ax1.imshow(spectra_before[:,:,330].T)
    ax1.set_title('spectra_before')
    ax2.imshow(spectra_after[:,:,330].T)
    ax2.set_title('spectra_after')
    ax3.imshow(region_before[:,:,330].T)
    ax3.set_title('extracted_region_before')
    ax4.imshow(region_after[:,:,330].T)
    ax4.set_title('extracted_region_after')
    # plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'cropped image example.png'))
    plt.show()
    st()

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]

    # Extract the subregion and wavelength range
    extracted_region_before = spectra_before[x_start:x_end, y_start:y_end, z_indices]
    extracted_region_after = spectra_after[x_start_1:x_end_1, y_start_1:y_end_1, z_indices]

    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_before = np.mean(extracted_region_before, axis=(0, 1))
    mean_spectrum_after = np.mean(extracted_region_after, axis=(0, 1))

    # Calculate 10^(-mean_spectrum_after)
    transformed_spectrum_before = 10 ** (-mean_spectrum_before)
    transformed_spectrum = 10 ** (-mean_spectrum_after)
    # transformed_spectrum_before = mean_spectrum_before
    # transformed_spectrum = mean_spectrum_after

    # Apply rubberband baseline correction to the transformed spectrum
    # corrected_spectrum_before, lower_baseline = rubberband_baseline_correction(wavelengths[z_indices], transformed_spectrum_before)
    # corrected_spectrum, lower_baseline = rubberband_baseline_correction(wavelengths[z_indices], transformed_spectrum)
    corrected_spectrum_before = transformed_spectrum_before
    corrected_spectrum = transformed_spectrum

    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths[z_indices], transformed_spectrum_before, label='A_before', color='y', linewidth=2)
    plt.plot(wavelengths[z_indices], transformed_spectrum, label='A_after', color='k', linewidth=2)
    # plt.plot(wavelengths[z_indices], corrected_spectrum_before, label='Baseline Corrected Spectrum before', color='r', linewidth=2)
    # plt.plot(wavelengths[z_indices], corrected_spectrum, label='Baseline Corrected Spectrum after', color='b', linewidth=2)
    # plt.plot(wavelengths[z_indices], corrected_spectrum, label='Baseline Corrected Spectrum', color='r', linewidth=2, linestyle='-.')
    # plt.plot(wavelengths[z_indices], baseline, label='Estimated Baseline', color='m', linewidth=2, linestyle=':')
    # plt.plot(wavelengths, flipped_spectrum, label='Flipped Spectrum', color='g', linestyle='--', linewidth=2)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Average Spectrum - local maximum Correction')
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig(os.path.join(save_path, 'spectral visualization.png'))
    # st()

    # Define your x ranges for maxima
    # x_ranges = [(1420, 1450), (1490,1510), (1570, 1580)]

    # upper_baseline, x_peaks = local_max_baseline_with_x_ranges(wavelengths[z_indices], corrected_spectrum, x_ranges, min_x_distance=10)
    # flipped_spectrum, final_flipped = flip_between_local_maxima(wavelengths[z_indices], corrected_spectrum, x_peaks, upper_baseline)


    # Cross-correlate the two spectra
    correlation = correlate(corrected_spectrum_before, corrected_spectrum, mode='full')
    lag = np.argmax(correlation) - (len(corrected_spectrum) - 1)

    # Convert lag to shift in terms of x-axis units (e.g., wavenumber, cm⁻¹)
    x_step = wavelengths[1] - wavelengths[0]  # Assumes equal spacing in x-axis values
    x_shift = lag * x_step

    print(f"The shift between the two spectra is approximately {x_shift:.2f} x-axis units.")
    # st()
    # st()
    ###right- left+
    sv = -30

    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths[z_indices], corrected_spectrum_before, label='Baseline Corrected Spectrum - before collagen', color='b', linewidth=2)
    plt.plot(wavelengths[z_indices], corrected_spectrum, label='Baseline Corrected Spectrum - after collagen', color='g', linewidth=2)
    plt.plot(wavelengths + x_shift - sv, corrected_spectrum, label=f'Spectrum shift back {x_shift:.0f}+{sv:.0f} units', color='g', linestyle='--', linewidth=2)
    
    corrected_spectrum_before[np.where(corrected_spectrum_before==0)] = 0.00001
    corrected_spectrum[np.where(corrected_spectrum==0)] = 0.00001

    corrected_spectrum_before_zeros = np.zeros((corrected_spectrum_before.shape[0]*2))
    corrected_spectrum_zeros = np.zeros((corrected_spectrum_before.shape[0]*2))


    corrected_spectrum_before_zeros[:corrected_spectrum_before.shape[0]] = corrected_spectrum_before
    corrected_spectrum_zeros[int(lag-sv/x_step):int(lag-sv/x_step + corrected_spectrum.shape[0])] = corrected_spectrum

    corrected_spectrum_before_zeros_copy = np.copy(corrected_spectrum_before_zeros)
    corrected_spectrum_zeros_copy = np.copy(corrected_spectrum_zeros)

    corrected_spectrum_before_zeros_copy[np.where(corrected_spectrum_before_zeros==0)]=0
    corrected_spectrum_zeros_copy[np.where(corrected_spectrum_zeros==0)]=0

    diff_ = corrected_spectrum_before_zeros_copy - corrected_spectrum_zeros_copy

    plt.plot(wavelengths[z_indices], diff_[:corrected_spectrum_before.shape[0]], label='Diff Spectrum', color='r', linewidth=2)


    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc="upper left")
    # plt.show()
    # st()
    plt.savefig(os.path.join(save_path, f'How much Spectrum Shifted.png'))

    output_filename = 'spectral_data_1.mat' 
    save_spectrum_to_mat(diff_[:corrected_spectrum_before.shape[0]], output_filename, save_path)

if __name__ == '__main__':
    main()