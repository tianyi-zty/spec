__author__ = 'Tianyi'

import spectrochempy as scp
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from scipy.io import savemat

###########################note###########################
# Perform rubberband baseline correction for a given spectrum.

def rubberband_baseline_correction(y, x=None):
    """
    Perform rubberband baseline correction for a given spectrum.

    Parameters:
        y (array-like): The y-axis values (e.g., intensity).
        x (array-like): The x-axis values (e.g., wavelength). If None, indices will be used.

    Returns:
        baseline (array-like): The estimated baseline.
        corrected_y (array-like): The baseline-corrected y values.
    """
    if x is None:
        x = np.arange(len(y))
    
    # Find the convex hull of the spectrum
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    
    # Get indices of the hull vertices, sort them
    hull_indices = np.sort(hull.vertices)
    
    # Filter hull indices to include only those at the baseline
    baseline_indices = hull_indices[np.diff(hull_indices, prepend=-1) != 1]  # Exclude upper hull points
    
    # Interpolate the baseline
    baseline = np.interp(x, x[baseline_indices], y[baseline_indices])
    
    # Correct the spectrum
    corrected_y = y - baseline
    
    return baseline, corrected_y

def local_max_baseline_with_x_ranges(x, y, x_ranges, min_x_distance=10):
    """
    Generate a baseline using local maxima within specified x-ranges by:
    - Extracting vertices of local maxima within the specified ranges.
    - Rotating to start from the lowest x-value.
    - Keeping only the ascending part of the maxima.
    - Interpolating a baseline using these vertices, stopping at the last local maximum.

    Parameters:
        x (array-like): The x-axis values (e.g., wavelength or wavenumber).
        y (array-like): The y-axis values (e.g., intensity).
        x_ranges (list of tuples): List of x-ranges (min_x, max_x) to restrict the search for local maxima.
        min_x_distance (int): Minimum x distance between peaks.

    Returns:
        baseline (array-like): The estimated baseline, stopping at the last local maximum.
    """
    # Calculate the minimum index distance based on min_x_distance
    min_index_distance = np.searchsorted(x, x[0] + min_x_distance) - 1

    # Initialize lists for x and y of local maxima within specified ranges
    x_peaks_list = []
    y_peaks_list = []
    
    for x_min, x_max in x_ranges:
        # Find indices within the current x-range
        range_indices = np.where((x >= x_min) & (x <= x_max))[0]
        if len(range_indices) == 0:
            continue  # Skip empty ranges
        
        # Find local maxima within the range with minimum x-distance constraint
        peaks, _ = find_peaks(y[range_indices], distance=min_index_distance)
        x_peaks = x[range_indices][peaks]
        y_peaks = y[range_indices][peaks]
        
        # Append these peaks to the lists
        x_peaks_list.extend(x_peaks)
        y_peaks_list.extend(y_peaks)
    
    # Sort the peaks by x value
    x_peaks = np.array(x_peaks_list)
    y_peaks = np.array(y_peaks_list)
    sorted_indices = np.argsort(x_peaks)
    x_peaks = x_peaks[sorted_indices]
    y_peaks = y_peaks[sorted_indices]
    # st()
    
    # Rotate the peaks to start from the lowest x-value
    min_index = np.argmin(x_peaks)
    x_peaks = np.roll(x_peaks, -min_index)
    y_peaks = np.roll(y_peaks, -min_index)
    print('local maximum find at:', x_peaks)
    # # Keep only the ascending part
    # ascending_indices = np.where(np.diff(y_peaks) >= 0)[0] + 1
    # x_peaks = x_peaks[:ascending_indices[-1] + 1]
    # y_peaks = y_peaks[:ascending_indices[-1] + 1]
    
    # Stop the baseline at the last local maximum by setting the x limit
    x_max_limit = x_peaks[-1]
    # st()
    
    # Interpolate the baseline between these vertices, stopping at the last local maximum
    # x_interp = x[x <= x_max_limit]
    upper_baseline = np.interp(x, x_peaks, y_peaks)
    
    # Find the index where x exceeds the last maximum
    beyond_max_index = np.where(x > x_max_limit)[0]
    
    # Replace baseline values beyond the last maximum with original y values
    if beyond_max_index.size > 0:
        upper_baseline[beyond_max_index] = y[beyond_max_index]
    # st()
    
    return upper_baseline, x_peaks


def flip_dips_to_new_convex_with_trend(y, x=None, corrected_baseline=None):
    """
    Flip dips below the line connecting local maxima to form new convex shapes, allowing the flipped spectrum 
    to indicate the trend (without replacing values below the baseline) and ending at the convex shape formed 
    by local maxima.
    
    Parameters:
        y (array-like): The y-axis values (e.g., intensity).
        x (array-like): The x-axis values (e.g., wavelength). If None, indices will be used.
        corrected_baseline (array-like): The corrected spectrum to ensure no spectrum falls below it.
        
    Returns:
        flipped_y (array-like): The spectrum with dips flipped into convex shapes, indicating the trend.
    """
    if x is None:
        x = np.arange(len(y))  # Use indices if x is not provided
    
    if corrected_baseline is None:
        raise ValueError("Corrected baseline must be provided to ensure the spectrum stays above it.")
    
    # Find local maxima (peaks)
    peaks, _ = find_peaks(y)
    
    # Include boundary points as maxima (first and last points)
    peaks = np.concatenate(([0], peaks, [len(y) - 1]))
    
    # Create a copy of the spectrum to store flipped results
    flipped_y = np.copy(y)
    
    # Iterate over adjacent maxima to form the convex shape
    for i in range(len(peaks) - 1):
        left_idx, right_idx = peaks[i], peaks[i + 1]
        
        # Linearly interpolate between two maxima
        x_left, x_right = x[left_idx], x[right_idx]
        y_left, y_right = y[left_idx], y[right_idx]
        
        # Line equation: y_line = m * (x - x_left) + y_left
        m = (y_right - y_left) / (x_right - x_left)  # Slope
        line_y = m * (x[left_idx:right_idx + 1] - x_left) + y_left
        
        # Flip dips to form a convex spectrum, but follow the trend without replacing below the baseline
        flipped_y[left_idx:right_idx + 1] = 2 * line_y - y[left_idx:right_idx + 1]
    
    # The final flipped spectrum follows the convex trend
    # Ensure the flipped spectrum ends at the local maxima convex shape
    for i in range(1, len(peaks)):
        left_idx, right_idx = peaks[i-1], peaks[i]
        flipped_y[left_idx:right_idx+1] = np.maximum(flipped_y[left_idx:right_idx+1], y[left_idx:right_idx+1])
    
    return flipped_y

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
    before_collagen = r'W:/3. Students/TianYi/Caf2_11082024/beforecollagen/LMT_1.mat'
    after_collagen = r'W:/3. Students/TianYi/Caf2_11152024/aftercollagen/LMT_1.mat'
    save_path = '../res/Caf2_11152024/rubberband'

    wavelength_start = 950
    wavelength_end = 1800
    wavelengths = np.linspace(950, 1800, 426)

    data_before = loadmat(before_collagen)
    spectra_before = np.reshape(data_before['r'], (480, 480, 426))
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))

    x_start, x_end = 100, 400
    y_start, y_end = 100, 400

    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]

    extracted_region_before = spectra_before[x_start:x_end, y_start:y_end, z_indices]
    extracted_region_after = spectra_after[x_start:x_end, y_start:y_end, z_indices]

    mean_spectrum_before = np.mean(extracted_region_before, axis=(0, 1))
    mean_spectrum_after = np.mean(extracted_region_after, axis=(0, 1))

    # Apply Rubberband baseline correction
    baseline_before, corrected_spectrum_before = rubberband_baseline_correction(mean_spectrum_before, wavelengths[z_indices])
    baseline_after, corrected_spectrum_after = rubberband_baseline_correction(mean_spectrum_after, wavelengths[z_indices])

    flipped_spectrum_after = flip_dips_to_new_convex_with_trend(mean_spectrum_after, x=wavelengths[z_indices], corrected_baseline=baseline_after)
    corrected_flipped_spectrum, corrected_flipped_spectrum_after = rubberband_baseline_correction(flipped_spectrum_after, wavelengths[z_indices])

    plt.figure(figsize=(20, 14))
    plt.plot(wavelengths[z_indices], mean_spectrum_after, label='Raw Spectrum - After', color='r')
    plt.plot(wavelengths[z_indices], baseline_after, label='Baseline - After', color='purple')
    plt.plot(wavelengths[z_indices], corrected_spectrum_after, label='Corrected Spectrum - After', color='k')
    plt.plot(wavelengths[z_indices], flipped_spectrum_after, label='Convex Spectrum - After (Dips Flipped)', color='b')
    plt.plot(wavelengths[z_indices], corrected_flipped_spectrum_after, label='Baseline corrected Convex Spectrum - After (Dips Flipped)', color='g')
    plt.xlabel('Wavelength (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Rubberband Baseline Correction')
    # plt.show()
    st()
    plt.savefig(os.path.join(save_path, 'Rubberband_Correction_Spectrum_After.png'))

    output_filename = 'Rubberband_flipped_spectrum_after.mat'
    save_spectrum_to_mat(flipped_spectrum_after, output_filename, save_path)


if __name__ == '__main__':
    main()
