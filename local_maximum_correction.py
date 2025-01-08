__author__ = 'Tianyi'

import spectrochempy as scp
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks



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
    baseline = np.interp(x, x[v], y[v])
    
    # Subtract the baseline from the original spectrum
    corrected_y = y - baseline
    
    return corrected_y, baseline

def flip_spectrum_section(x, y, upper_x, lower_x, upper_peak, lower_peak):
    """
    Flip the section of the spectrum between the upper and lower peaks along the line connecting them.

    Parameters:
        x (array-like): Wavelength or wavenumber values.
        y (array-like): Intensity values.
        upper_x, lower_x (float): x values of the upper and lower peaks.
        upper_peak, lower_peak (float): Intensity values of the upper and lower peaks.

    Returns:
        flipped_y (array-like): The intensity values after flipping the selected segment.
        line_x (array-like): The x-values for the line connecting the peaks.
        line_y (array-like): The y-values for the line connecting the peaks.
    """
    # Make a copy of the original spectrum to store the flipped values
    flipped_y = np.copy(y)

    # Calculate the slope and intercept of the line connecting the two peaks
    slope = (lower_peak - upper_peak) / (lower_x - upper_x)
    intercept = upper_peak - slope * upper_x

    # Define the line segment only between the two peaks
    line_x = np.array([upper_x, lower_x])
    line_y = np.array([upper_peak, lower_peak])
    # st()

    # Create a mask to identify points in the x range between the peaks
    mask = (x >= min(lower_x, upper_x)) & (x <= max(lower_x, upper_x))

    # Select the x and y values in the region to flip
    x_segment = x[mask]
    y_segment = y[mask]

    # Calculate the y values on the line at each x in the segment
    y_on_line = slope * x_segment + intercept

    # Flip the y values in the segment by reflecting them along y_on_line
    flipped_y[mask] = 2 * y_on_line - y_segment
    final_flipped = y_on_line - y_segment
    # st()
    return flipped_y, line_x, line_y, x_segment, final_flipped

def find_local_maximum_in_range(x, y, x_range):
    """
    Find one local maximum within a specified x range.

    Parameters:
        x (array-like): Wavelength or wavenumber.
        y (array-like): Intensity values.
        x_range (tuple): The range of x values (min_x, max_x).

    Returns:
        peak (tuple): (x-value, intensity) of the local maximum found in the range.
    """
    
    # Filter the data within the specified range
    mask = (x >= x_range[0]) & (x <= x_range[1])
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Find all local maxima
    peaks, _ = find_peaks(y_filtered)

    if len(peaks) == 0:
        raise ValueError(f"No peaks found in the range {x_range}.")

    # Find the highest peak in the filtered range
    peak_index = peaks[np.argmax(y_filtered[peaks])]
    return x_filtered[peak_index], y_filtered[peak_index]

# def find_local_maxima(x, y, x_range, min_distance=10):
#     """
#     Find the indices of the local maxima within a specified x range.

#     Parameters:
#         x (array-like): Wavelength or wavenumber.
#         y (array-like): Intensity values.
#         x_range (tuple): The range of x values for which to find the maxima (min_x, max_x).

#     Returns:
#         upper_hull (array-like): Indices of the upper convex hull.
#         lower_hull (array-like): Indices of the lower convex hull.
#     """
#     # Filter the data within the specified range
#     mask = (x >= x_range[0]) & (x <= x_range[1])
#     x_filtered = x[mask]
#     y_filtered = y[mask]

#     # Find all local maxima
#     peaks, _ = find_peaks(y_filtered)
#     # st()
    
#     # Check if we have enough peaks
#     if len(peaks) < 2:
#         raise ValueError("Not enough peaks to form a pair within the specified range.")
    
#     # st()
#     # Find pairs of peaks that satisfy the minimum distance criterion
#     for i in range(len(peaks)):
#         for j in range(i + 1, len(peaks)):
#             if abs(x_filtered[peaks[i]] - x_filtered[peaks[j]]) >= min_distance:
#                 upper_index = peaks[i] if y_filtered[peaks[i]] > y_filtered[peaks[j]] else peaks[j]
#                 lower_index = peaks[j] if upper_index == peaks[i] else peaks[i]
#                 upper_x, lower_x = x_filtered[upper_index], x_filtered[lower_index]
#                 upper_peak, lower_peak = y_filtered[upper_index], y_filtered[lower_index]

#                 # Print the x values of the pair of local maxima
#                 print(f"Upper Peak at x = {upper_x}, Lower Peak at x = {lower_x}")
#                 return upper_x, lower_x, upper_peak, lower_peak
#     st()
#     # If no pair meets the minimum distance, raise an error
#     raise ValueError("No peak pairs found with the specified minimum distance.")


def main():

    before_collagen = r'../data/AuPillars_10212024/beforeCollagen/2.mat'
    after_collagen = r'../data/AuPillars_10212024/afterCollagen/1.mat'
    save_path = '../res/AuPillars_10212024/'

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

    # Define the region of interest on the x and y axes
    x_start, x_end = 250, 350 #30, 230 #250, 350 #  Replace with your desired x range
    y_start, y_end = 160, 360  # Replace with your desired y range

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]

    # Extract the subregion and wavelength range
    extracted_region_before = spectra_before[x_start:x_end, y_start:y_end, z_indices]
    extracted_region_after = spectra_after[x_start:x_end, y_start:y_end, z_indices]

    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_before = np.mean(extracted_region_before, axis=(0, 1))
    mean_spectrum_after = np.mean(extracted_region_after, axis=(0, 1))

    # Calculate 10^(-mean_spectrum_after)
    transformed_spectrum_before = 10 ** (-mean_spectrum_before)
    transformed_spectrum = 10 ** (-mean_spectrum_after)

    # Apply rubberband baseline correction to the transformed spectrum
    corrected_spectrum, baseline = rubberband_baseline_correction(wavelengths[z_indices], transformed_spectrum)

    # Define the range for finding the upper and lower hulls
    # Define your x ranges for the upper and lower maxima
    upper_x_range = (1450, 1550)  # Replace with actual values
    lower_x_range = (1650, 1700)  # Replace with actual values

    # Find the upper and lower maxima within the specified ranges
    upper_x, upper_peak = find_local_maximum_in_range(wavelengths, corrected_spectrum, upper_x_range)
    lower_x, lower_peak = find_local_maximum_in_range(wavelengths, corrected_spectrum, lower_x_range)

    # Flip the spectrum between the two maxima
    flipped_spectrum, line_x, line_y, x_segment, final_flipped = flip_spectrum_section(wavelengths, corrected_spectrum, upper_x, lower_x, upper_peak, lower_peak)

    plt.figure(figsize=(12, 8))
    # plt.plot(wavelengths[z_indices], mean_spectrum_before, label='A_before', color='y', linewidth=2)
    # plt.plot(wavelengths[z_indices], mean_spectrum_after, label='A_after', color='k', linewidth=2)
    # plt.plot(wavelengths[z_indices], transformed_spectrum_before, label='R/R0 before', color='y', linewidth=2)
    # plt.plot(wavelengths[z_indices], transformed_spectrum, label='R/R0 after', color='b', linewidth=2)
    plt.plot(wavelengths[z_indices], corrected_spectrum, label='Baseline Corrected Spectrum', color='r', linewidth=2, linestyle='-.')
    # plt.plot(wavelengths[z_indices], baseline, label='Estimated Baseline', color='m', linewidth=2, linestyle=':')
    plt.plot(wavelengths, flipped_spectrum, label='Flipped Spectrum', color='g', linestyle='--', linewidth=2)
    plt.plot(line_x, line_y, label='Line Between Peaks', color='k', linestyle=':', linewidth=1)
    # Plot the transformed dashed line
    # plt.plot(x_segment,final_flipped, label='Final flipped A', color='k', linestyle=':', linewidth=1)
    plt.scatter(line_x, line_y, color='r')  # Mark the peaks

    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Average Spectrum - local maximum Correction')
    plt.legend(loc="upper left")
    plt.show()
    st()


if __name__ == '__main__':
    main()