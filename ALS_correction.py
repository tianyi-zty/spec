__author__ = 'Tianyi'
import os
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
from scipy.signal import find_peaks
from scipy.io import savemat

###########################note###########################
# Perform Asymmetric Least Squares (ALS) baseline correction for a given spectrum.

def als_baseline_correction(y, lam=1e6, p=0.01, n_iter=10):
    """
    Perform Asymmetric Least Squares (ALS) baseline correction for a given spectrum.
    
    Parameters:
        y (array-like): The y-axis values (e.g., intensity).
        lam (float): Smoothing factor (higher values create smoother baselines).
        p (float): Asymmetry parameter (0 < p < 1; smaller values give more weight to negative deviations).
        n_iter (int): Number of iterations to perform.

    Returns:
        baseline (array-like): The estimated baseline.
        corrected_y (array-like): The baseline-corrected y values.
    """
    y = np.asarray(y)  # Ensure y is a NumPy array
    L = len(y)  # Length of the spectrum

    # Second-order difference matrix of size (L-2, L)
    D = np.diff(np.eye(L), 2)
    
    # Compute D.T @ D and pad to size (L, L)
    DTD = D.T @ D
    DTD_padded = np.zeros((L, L))
    DTD_padded[:DTD.shape[0], :DTD.shape[1]] = DTD  # Pad with zeros to (L, L)

    # Initialize weights
    W = np.ones(L)  # Weights vector of shape (L,)
    
    for i in range(n_iter):
        # Create the diagonal weight matrix of shape (L, L)
        W_matrix = np.diag(W)
        
        # Add smoothness term (lam * DTD_padded) to weight matrix
        smooth_term = lam * DTD_padded
        
        # # Validate shapes (for debugging purposes)
        # print(f"Iteration {i+1}:")
        # print(f"  W_matrix shape: {W_matrix.shape}")
        # print(f"  D.T @ D shape (padded): {smooth_term.shape}")
        # print(f"  y shape: {y.shape}")
        
        # Compute the solution Z for the baseline
        Z = np.linalg.inv(W_matrix + smooth_term) @ (W_matrix @ y)
        
        # Update weights: smaller weights for points above the baseline
        W = p * (y > Z) + (1 - p) * (y <= Z)
    
    # The estimated baseline
    baseline = Z
    
    # Correct the spectrum
    corrected_y = y - baseline
    
    return baseline, corrected_y

def flip_dips_to_new_convex(y, x=None, corrected_baseline=None):
    """
    Flip dips below the line connecting local maxima to form new convex shapes, ensuring that the spectrum
    remains above the corrected baseline (i.e., `corrected_spectrum_after`).
    
    Parameters:
        y (array-like): The y-axis values (e.g., intensity).
        x (array-like): The x-axis values (e.g., wavelength). If None, indices will be used.
        corrected_baseline (array-like): The corrected spectrum to ensure no spectrum falls below it.
        
    Returns:
        flipped_y (array-like): The spectrum with dips flipped into convex shapes, above the corrected baseline.
    """
    if x is None:
        x = np.arange(len(y))  # Use indices if x is not provided
    
    if corrected_baseline is None:
        raise ValueError("Corrected baseline must be provided to ensure the spectrum stays above it.")
    
    # Find local maxima
    peaks, _ = find_peaks(y)
    
    # Include boundary points as maxima
    peaks = np.concatenate(([0], peaks, [len(y) - 1]))
    
    # Create a copy of the spectrum to store flipped results
    flipped_y = np.copy(y)
    
    # Iterate over adjacent maxima
    for i in range(len(peaks) - 1):
        left_idx, right_idx = peaks[i], peaks[i + 1]
        
        # Linearly interpolate between two maxima
        x_left, x_right = x[left_idx], x[right_idx]
        y_left, y_right = y[left_idx], y[right_idx]
        
        # Line equation: y_line = m * (x - x_left) + y_left
        m = (y_right - y_left) / (x_right - x_left)  # Slope
        line_y = m * (x[left_idx:right_idx + 1] - x_left) + y_left
        
        # Flip dips below the line to convex, but above the corrected baseline
        flipped_y[left_idx:right_idx + 1] = np.maximum(2 * line_y - y[left_idx:right_idx + 1], corrected_baseline[left_idx:right_idx + 1])
    
    return flipped_y

def flip_dips_to_new_convex_with_trend(y, x=None):
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

    # Find local maxima (peaks)
    peaks, _ = find_peaks(y, distance=10)
    # st()
    wavelengths = 950 + peaks * 2
    print("peaks are found at:", wavelengths, "nm", peaks,'index')
    # st()
    # Define the ranges of interest (start, end in nm)
    # ranges = [(210,220), (240,260), (270,280), (320,330), (390,410)]
    # ranges = [(175,185), (210,205), (260,275)]
    ranges = [(175,185), (210,205), (260,275)]


    filtered_peaks = []

    for a,b in ranges:
        filtered_peaks.extend([peak for peak in peaks if a<= peak <=b])

    print('filtered_peaks:',filtered_peaks, 'index')
    # st()
    # Include boundary points as maxima (first and last points)
    # peaks = np.concatenate(([0], filtered_peaks, [len(y) - 1]))
    peaks = filtered_peaks
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

    before_collagen = r'W:/3. Students/Tianyi/AuPillars_10nmAl2O3_12102024/99-1/after/LMR_1.mat'
    after_collagen = r'W:/3. Students/Tianyi/AuPillars_10nmAl2O3_12102024/95-5/after/LMR_1.mat'
    # collagen_ref = r'W:/3. Students/TianYi/AuPillars_11042024/afterCollagen/LMT_2.mat'
    save_path = '../res/AuPillars_Al2O3_12102024/ALS/1/'

    # Wavelength range (cm⁻¹)
    wavelength_start = 950
    wavelength_end = 1800
    wavelength_start_plot = 1400
    wavelength_end_plot =1700
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    # Load the .mat file
    data_before = loadmat(before_collagen)
    spectra_before = np.reshape(data_before['r'], (480, 480, 426))
    data_after = loadmat(after_collagen)
    spectra_after = np.reshape(data_after['r'], (480, 480, 426))
    # data_ref = loadmat(collagen_ref)
    # spectra_ref = np.reshape(data_ref['r'], (480, 480, 426))
    # st()
    # Define the region of interest on the x and y axes
    x1_start, x1_end = 80, 200 #280, 400 #80, 200 # #270, 420   #250, 350 # 30, 230 # # Replace with your desired y range
    y1_start, y1_end = 80, 200 #80, 200 #150, 300  # Replace with your desired x range
    # x2_start, x2_end = 280, 400
    # y2_start, y2_end = 80, 200

    # Find the indices corresponding to the desired wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    z_indices_plot = np.where((wavelengths >= wavelength_start_plot) & (wavelengths <= wavelength_end_plot))[0]
    # Extract the subregion and wavelength range
    extracted_region_before = spectra_before[x1_start:x1_end, y1_start:y1_end, z_indices]
    extracted_region_after = spectra_after[x1_start:x1_end, y1_start:y1_end, z_indices]
    # extracted_region_ref = spectra_ref[r1_start:r1_end, r2_start:r2_end, z_indices]
    # st()
    # Plot the results
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
    # (ax1, ax2), (ax3, ax4) = axes  # Unpack axes for easier reference
    # ax1.imshow(spectra_before[:,:,330])
    # ax1.set_title('spectra_before')
    # ax2.imshow(spectra_after[:,:,330])
    # ax2.set_title('spectra_after')
    # ax3.imshow(extracted_region_before[:,:,330])
    # ax3.set_title('extracted_region_before')
    # ax4.imshow(extracted_region_after[:,:,330])
    # ax4.set_title('extracted_region_after')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(os.path.join(save_path, 'cropped image example.png'))
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_before = np.mean(extracted_region_before, axis=(0, 1))
    mean_spectrum_after = np.mean(extracted_region_after, axis=(0, 1))
    # mean_spectrum_ref = np.mean(extracted_region_ref, axis=(0, 1))

    # Calculate 10^(-mean_spectrum_after)
    transformed_spectrum_before = 10 ** (-mean_spectrum_before)
    transformed_spectrum_after = 10 ** (-mean_spectrum_after)

    # Apply a ALS baseline correction
    lam = 1e6  # Adjust as needed
    p = 0.01   # Adjust as needed
    baseline_before, corrected_spectrum_before = als_baseline_correction(transformed_spectrum_before, lam=lam, p=p)
    baseline_after, corrected_spectrum_after = als_baseline_correction(transformed_spectrum_after, lam=lam, p=p)

    # Flip dips to form a convex spectrum
    # flipped_spectrum_after = flip_dips_to_new_convex(corrected_spectrum_after, x=wavelengths[z_indices], corrected_baseline=corrected_spectrum_after)

    # Flip dips to form a convex spectrum, following the trend
    flipped_spectrum_before = flip_dips_to_new_convex_with_trend(corrected_spectrum_before, x=wavelengths[z_indices])
    flipped_spectrum_after = flip_dips_to_new_convex_with_trend(corrected_spectrum_after, x=wavelengths[z_indices])

    corrected_flipped_spectrum, corrected_flipped_spectrum_before = als_baseline_correction(flipped_spectrum_before, lam=lam, p=p)
    corrected_flipped_spectrum, corrected_flipped_spectrum_after = als_baseline_correction(flipped_spectrum_after, lam=lam, p=p)

    # # Plot the results
    plt.figure(figsize=(20, 14))
    # plt.plot(wavelengths[z_indices], mean_spectrum_after, label='Raw Spectrum - After', color='r')
    # plt.plot(wavelengths[z_indices], baseline_after, label='Baseline - After', color='purple')
    plt.plot(wavelengths[z_indices], corrected_spectrum_before, label='Corrected Spectrum - 99-1', color='k', linestyle='--')
    plt.plot(wavelengths[z_indices], corrected_spectrum_after, label='Corrected Spectrum - 95-5', color='k')
    plt.xlabel('Wavelength (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Asymmetric Least Squares Baseline Correction')
    # plt.show()
    plt.savefig(os.path.join(save_path, 'ALS_correction Spectrum - 99-1 & 95-5.png'))

    plt.figure(figsize=(20, 14))
    plt.plot(wavelengths[z_indices], corrected_spectrum_before, label='Corrected Spectrum - Before', color='k', linestyle='--')
    plt.plot(wavelengths[z_indices], flipped_spectrum_before, label='Convex Spectrum - Before (Dips Flipped)', color='b', linestyle='--')
    # plt.plot(wavelengths[z_indices], corrected_flipped_spectrum_before, label='Baseline corrected Convex Spectrum - Before (Dips Flipped)', color='g', linestyle='--',linewidth=5.0)
    plt.xlabel('Wavelength (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Asymmetric Least Squares Baseline Correction 99-1')
    # plt.show()
    plt.savefig(os.path.join(save_path, 'ALS_correction Spectrum - 99-1.png'))

    plt.figure(figsize=(20, 14))
    plt.plot(wavelengths[z_indices], corrected_spectrum_after, label='Corrected Spectrum - After', color='k')
    plt.plot(wavelengths[z_indices], flipped_spectrum_after, label='Convex Spectrum - After (Dips Flipped)', color='b')
    # plt.plot(wavelengths[z_indices], corrected_flipped_spectrum_after, label='Baseline corrected Convex Spectrum - After (Dips Flipped)', color='g',linewidth=5.0)
    plt.xlabel('Wavelength (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Asymmetric Least Squares Baseline Correction 95-5')
    # plt.show()
    plt.savefig(os.path.join(save_path, 'ALS_correction Spectrum - 95-5.png'))

    plt.figure(figsize=(20, 14))
    plt.plot(wavelengths[z_indices_plot], corrected_spectrum_before[z_indices_plot], label='Corrected Spectrum - After', color='k')
    plt.plot(wavelengths[z_indices_plot], corrected_spectrum_after[z_indices_plot], label='Convex Spectrum - After (Dips Flipped)', color='b')
    # plt.plot(wavelengths[z_indices], corrected_flipped_spectrum_after, label='Baseline corrected Convex Spectrum - After (Dips Flipped)', color='g',linewidth=5.0)
    plt.xlabel('Wavelength (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Asymmetric Least Squares Baseline Correction 95-5')
    # plt.show()
    plt.savefig(os.path.join(save_path, 'ROI Spectrum.png'))

    output_filename = 'ALS_corrected_flipped_spectrum_99-1.mat' 
    save_spectrum_to_mat(corrected_flipped_spectrum_before, output_filename, save_path)
    output_filename = 'ALS_corrected_flipped_spectrum_95-5.mat' 
    save_spectrum_to_mat(corrected_flipped_spectrum_after, output_filename, save_path)


if __name__ == '__main__':
    main()