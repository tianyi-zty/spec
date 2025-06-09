import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
from scipy.ndimage import gaussian_filter1d

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

def save_spectrum_to_mat(spectrum, filename, save_path):
    """
    Save the spectrum to a .mat file at the specified path.
    
    Parameters:
        spectrum (array-like): The spectrum to save.
        filename (str): The name of the output .mat file.
        save_path (str): The directory path where the .mat file should be saved.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the directory if it doesn't exist
    
    full_path = os.path.join(save_path, filename)
    data_dict = {'spectrum': spectrum}
    savemat(full_path, data_dict)
    print(f"Spectrum saved to {full_path}")

def main():
        # Plot the average spectrum with standard deviation
        plt.figure(figsize=(12, 8))

        path = r'../res/AuPillars_50nmAl2O3_2_05222025/1'

        for file in os.listdir(path):
            if file.endswith(".mat") and not file.endswith("smooth.mat"):
                mat_path = os.path.join(path, file)
                # st()
                name = file.split('.mat')[0]
                wavelengths = np.linspace(950, 1800, 426)
                data_after = loadmat(mat_path)
                spectra_after = np.reshape(data_after['spectrum'], (426))
                # Apply Gaussian smoothing
                sigma = 3  # you can adjust this value
                smoothed_spectrum = gaussian_filter1d(spectra_after, sigma=sigma)
                # Save the subspectrum
                subspectrum_filename = f"{name}_gaussian_3_smooth.mat"
                savemat(os.path.join(path, subspectrum_filename), {'spectrum': smoothed_spectrum})
                # # Apply a ALS baseline correction
                # lam = 1e6  # Adjust as needed
                # p = 0.01   # Adjust as needed
                # baseline_before, corrected_smoothed_spectrum = als_baseline_correction(smoothed_spectrum, lam=lam, p=p)
                plt.plot(wavelengths, smoothed_spectrum, label=f'{name}_gaussian_3_smooth', linewidth=2)
                # plt.plot(wavelengths, spectra_after, label=f'{name}',linewidth=2) 
                plt.xlabel('Wavenumber (cm⁻¹)')
                plt.ylabel('Intensity')
                plt.legend(loc='upper left')
                plt.title(f"Spectrum")
                plt.grid(True)

        # plt.show()
        plt.savefig(os.path.join(path, 'spectrum visualization after smooth.png'))
        st()
            # Save


if __name__ == '__main__':
    main()