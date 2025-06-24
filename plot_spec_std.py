import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
import glob



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


def main():
        # Plot the average spectrum with standard deviation
        plt.figure(figsize=(12, 8))

        path = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/1000'
        save_path = r'../res/caf2_06132025/tnse/result/3group/original/'
        file_list = sorted(glob.glob(os.path.join(path, '*.npy')))
        # Plot all spectra
        plt.figure(figsize=(10, 6))

        for file_path in file_list:
            spectrum = np.load(file_path)  # shape should be (426,)
            if spectrum.ndim == 1:
                plt.plot(spectrum, alpha=0.3, linewidth=0.7)  # transparent for overlap
            else:
                print(f"Skipping {file_path}, not 1D shape: {spectrum.shape}")

        plt.title("Spectra from '1000/' folder")
        plt.xlabel("Wavenumber index")
        plt.ylabel("Intensity (normalized)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'1000.png'))
        plt.show()

        path1 = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/8020/'
        file_list1 = sorted(glob.glob(os.path.join(path1, '*.npy')))
        # Plot all spectra
        plt.figure(figsize=(10, 6))

        for file_path in file_list1:
            spectrum = np.load(file_path)  # shape should be (426,)
            if spectrum.ndim == 1:
                plt.plot(spectrum, alpha=0.3, linewidth=0.7)  # transparent for overlap
            else:
                print(f"Skipping {file_path}, not 1D shape: {spectrum.shape}")

        plt.title("Spectra from '8020/' folder")
        plt.xlabel("Wavenumber index")
        plt.ylabel("Intensity (normalized)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'8020.png'))
        plt.show()
        
        path1 = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/6040/'
        file_list1 = sorted(glob.glob(os.path.join(path1, '*.npy')))
        # Plot all spectra
        plt.figure(figsize=(10, 6))

        for file_path in file_list1:
            spectrum = np.load(file_path)  # shape should be (426,)
            if spectrum.ndim == 1:
                plt.plot(spectrum, alpha=0.3, linewidth=0.7)  # transparent for overlap
            else:
                print(f"Skipping {file_path}, not 1D shape: {spectrum.shape}")

        plt.title("Spectra from '6040/' folder")
        plt.xlabel("Wavenumber index")
        plt.ylabel("Intensity (normalized)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'6040.png'))
        plt.show()

        # path_1 = r'../res/caf2_06132025/bgcorrect/1000/LMT_4'
        # col_data_1 = loadmat(os.path.join(path_1, "resonance_LMT_4_after_mask1.mat"))
        # # base_1 = loadmat(os.path.join(path_1, "resonance_LMT_2_after_mask0_gaussian_3_smooth.mat"))
        # spectra_col_1 = np.reshape(col_data_1['spectrum'], (426))
        # # spectra_base_1 = np.reshape(base_1['spectrum'], (426))
        # # col_no_bg_1 = spectra_col_1 - spectra_base_1
        # baseline_1,spectra_col_als_1 = als_baseline_correction(spectra_col_1)

        # # path_2 = r'../res/caf2_06132025/bgcorrect/8020/LMT_4'
        # # col_data_2 = loadmat(os.path.join(path_2, "resonance_LMT_4_after_mask1.mat"))
        # # # base_2 = loadmat(os.path.join(path_2, "resonance_LMT_2_after_mask0_gaussian_3_smooth.mat"))
        # # spectra_col_2 = np.reshape(col_data_2['spectrum'], (426))
        # # # spectra_base_2 = np.reshape(base_2['spectrum'], (426))
        # # # col_no_bg_2 = spectra_col_2 - spectra_base_2
        # # baseline_2, spectra_col_als_2 = als_baseline_correction(spectra_col_2)

        # # path_3 = r'../res/caf2_06132025/bgcorrect/8020/LMT_5'
        # # col_data_3 = loadmat(os.path.join(path_3, "resonance_LMT_5_after_mask1.mat"))
        # # # base_3 = loadmat(os.path.join(path_3, "resonance_LMT_1_after_mask0_gaussian_3_smooth.mat"))
        # # spectra_col_3 = np.reshape(col_data_3['spectrum'], (426))
        # # # spectra_base_3 = np.reshape(base_3['spectrum'], (426))
        # # # col_no_bg_3 = spectra_col_3 - spectra_base_3
        # # baseline_3, spectra_col_als_3 = als_baseline_correction(spectra_col_3)

        # wavelengths = np.linspace(950, 1800, 426)
        # plt.plot(wavelengths, spectra_col_als, label='1000LMT_3',linewidth=2) 
        # plt.plot(wavelengths, spectra_col_als_1, label='1000LMT_4',linewidth=2) 
        # plt.plot(wavelengths, spectra_col_als_2, label='8020LMT_4',linewidth=2) 
        # plt.plot(wavelengths, spectra_col_als_3, label='8020LMT_5',linewidth=2) 
        # plt.xlabel('Wavenumber (cm⁻¹)')
        # plt.ylabel('Intensity')
        # plt.legend(loc='upper left')
        # plt.title(f"Spectrum")
        # plt.grid(True)

        # plt.show()
        # # plt.savefig(os.path.join(path, 'spectrum visualization.png'))


        # # for file in os.listdir(path):
        # #     if file.endswith(".mat")and not file.endswith(".mat"):
        # #         mat_path = os.path.join(path, file)
        # #         # st()
        # #         name = file.split('.mat')[0]
        # #         wavelengths = np.linspace(950, 1800, 426)
        # #         data_after = loadmat(mat_path)
        # #         spectra_after = np.reshape(data_after['spectrum'], (426))
        # #         plt.plot(wavelengths, spectra_after, label=f'{name}',linewidth=2) 
        # #         plt.xlabel('Wavenumber (cm⁻¹)')
        # #         plt.ylabel('Intensity')
        # #         plt.legend(loc='upper left')
        # #         plt.title(f"Spectrum")
        # #         plt.grid(True)

        # # # plt.show()
        # # plt.savefig(os.path.join(path, 'spectrum visualization.png'))
        #     # Save


if __name__ == '__main__':
    main()