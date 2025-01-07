__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import correlate



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

def flip_between_local_maxima(x, y, x_peaks, baseline):
    """
    Flip sections of the spectrum according to the baseline between each pair of local maxima.
    """
    y_flipped = np.copy(y)
    final_flipped = np.copy(y)
    for i in range(len(x_peaks) - 1):
        x_min = x_peaks[i]
        x_max = x_peaks[i + 1]
        flip_indices = np.where((x >= x_min) & (x <= x_max))[0]
        y_flipped[flip_indices] = 2 * baseline[flip_indices] - y[flip_indices]
        final_flipped[flip_indices] = baseline[flip_indices] - y[flip_indices]
    # st()
    # Set `final_flipped` to 0 beyond the last maximum
    if len(x_peaks) > 0:
        last_max_index = np.where(x > x_peaks[-1])[0]
        final_flipped[last_max_index] = 0

    return y_flipped, final_flipped

def main():

    before_collagen = r'W:/3. Students/TianYi/AuPillars_11042024/beforeCollagen/LMR_4.mat'
    after_collagen = r'W:/3. Students/TianYi/AuPillars_11042024/afterCollagen/LMR_4.mat'
    save_path = '../res/AuPillars_11042024/1000-1100'

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
    x_start, x_end = 250, 350 #30, 230 #250, 350 # Replace with your desired x range
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
    corrected_spectrum_before, lower_baseline = rubberband_baseline_correction(wavelengths[z_indices], transformed_spectrum_before)
    corrected_spectrum, lower_baseline = rubberband_baseline_correction(wavelengths[z_indices], transformed_spectrum)

    # Define your x ranges for maxima
    x_ranges = [(1420, 1450), (1480, 1490), (1560,1600), (1670, 1680)]

    upper_baseline, x_peaks = local_max_baseline_with_x_ranges(wavelengths[z_indices], corrected_spectrum, x_ranges, min_x_distance=10)
    flipped_spectrum, final_flipped = flip_between_local_maxima(wavelengths[z_indices], corrected_spectrum, x_peaks, upper_baseline)

    # Cross-correlate the two spectra
    correlation = correlate(corrected_spectrum_before, flipped_spectrum, mode='full')
    lag = np.argmax(correlation) - (len(flipped_spectrum) - 1)

    # Convert lag to shift in terms of x-axis units (e.g., wavenumber, cm⁻¹)
    x_step = wavelengths[1] - wavelengths[0]  # Assumes equal spacing in x-axis values
    x_shift = lag * x_step

    print(f"The shift between the two spectra is approximately {x_shift:.2f} x-axis units.")
    # st()

    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths[z_indices], corrected_spectrum_before, label='Baseline Corrected Spectrum - before collagen', color='b', linewidth=2)
    plt.plot(wavelengths, flipped_spectrum, label='Flipped Spectrum', color='g', linewidth=2)
    plt.plot(wavelengths + x_shift, flipped_spectrum, label=f'Flipped Spectrum shift back {x_shift:.2f} units', color='g', linestyle='--', linewidth=2)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc="upper left")
    # plt.show()
    # st()

    plt.savefig(os.path.join(save_path, f'How much Spectrum Shifted.png'))





if __name__ == '__main__':
    main()