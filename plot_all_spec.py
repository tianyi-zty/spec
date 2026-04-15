import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st
import glob
from scipy.spatial import ConvexHull


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

def rubberband_baseline_correction(x, y):
    """
    Rubberband baseline correction using the convex hull.
    
    Parameters:
        x (array-like): The x-axis values (e.g., wavenumber).
        y (array-like): The y-axis values (e.g., intensity).

    Returns:
        baseline (array): The rubberband baseline.
        corrected_y (array): The baseline-corrected spectrum.
    """
    x = np.array(x)
    y = np.array(y)

    # Get points forming the convex hull
    v = np.vstack((x, y)).T
    hull = ConvexHull(v)

    # Extract lower convex hull indices (start and end inclusive)
    hull_indices = sorted(hull.vertices)
    lower_indices = [idx for idx in hull_indices if idx == 0 or idx == len(x) - 1 or (y[idx] < y[idx-1] and y[idx] < y[idx+1])]
    lower_indices = np.array(sorted(lower_indices))

    # Interpolate baseline across those points
    baseline = np.interp(x, x[lower_indices], y[lower_indices])

    corrected_y = y - baseline

    return baseline, corrected_y


def main():
        # # Plot the average spectrum with standard deviation
        # plt.figure(figsize=(12, 8))

        # path = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/1000'
        # save_path = r'../res/caf2_06132025/tnse/result/3group/original/'
        # file_list = sorted(glob.glob(os.path.join(path, '*.npy')))
        # # Plot all spectra
        # plt.figure(figsize=(10, 6))

        # for file_path in file_list:
        #     spectrum = np.load(file_path)  # shape should be (426,)
        #     if spectrum.ndim == 1:
        #         plt.plot(spectrum, alpha=0.3, linewidth=0.7)  # transparent for overlap
        #     else:
        #         print(f"Skipping {file_path}, not 1D shape: {spectrum.shape}")

        # plt.title("Spectra from '1000/' folder")
        # plt.xlabel("Wavenumber index")
        # plt.ylabel("Intensity (normalized)")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_path, f'1000.png'))
        # plt.show()

        # path1 = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/8020/'
        # file_list1 = sorted(glob.glob(os.path.join(path1, '*.npy')))
        # # Plot all spectra
        # plt.figure(figsize=(10, 6))

        # for file_path in file_list1:
        #     spectrum = np.load(file_path)  # shape should be (426,)
        #     if spectrum.ndim == 1:
        #         plt.plot(spectrum, alpha=0.3, linewidth=0.7)  # transparent for overlap
        #     else:
        #         print(f"Skipping {file_path}, not 1D shape: {spectrum.shape}")

        # plt.title("Spectra from '8020/' folder")
        # plt.xlabel("Wavenumber index")
        # plt.ylabel("Intensity (normalized)")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_path, f'8020.png'))
        # plt.show()
        
        # path1 = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/6040/'
        # file_list1 = sorted(glob.glob(os.path.join(path1, '*.npy')))
        # # Plot all spectra
        # plt.figure(figsize=(10, 6))

        # for file_path in file_list1:
        #     spectrum = np.load(file_path)  # shape should be (426,)
        #     if spectrum.ndim == 1:
        #         plt.plot(spectrum, alpha=0.3, linewidth=0.7)  # transparent for overlap
        #     else:
        #         print(f"Skipping {file_path}, not 1D shape: {spectrum.shape}")

        # plt.title("Spectra from '6040/' folder")
        # plt.xlabel("Wavenumber index")
        # plt.ylabel("Intensity (normalized)")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_path, f'6040.png'))
        # plt.show()

        wavelengths = np.linspace(950, 1800, 426)
        # path_1 = r'../res/rat/liver_ffpe/'
        # col_data_1 = loadmat(os.path.join(path_1, "HMT_5_after_mask1.mat"))
        # spectra_col_1 = np.reshape(col_data_1['spectrum'], (426))
        # baseline_1,spectra_col_als_1 = rubberband_baseline_correction(wavelengths,spectra_col_1)

        # path_2 = r'../res/rat/kidney_ffpe/'
        # col_data_2 = loadmat(os.path.join(path_2, "HMT_5_after_mask1.mat"))
        # spectra_col_2 = np.reshape(col_data_2['spectrum'], (426))
        # baseline_2, spectra_col_als_2 = rubberband_baseline_correction(wavelengths,spectra_col_2)

        # path_3 = r'../res/rat/liver_oct/'
        # col_data_3 = loadmat(os.path.join(path_3, "HMT_5_after_mask1.mat"))
        # spectra_col_3 = np.reshape(col_data_3['spectrum'], (426))
        # baseline_1,spectra_col_als_3 = rubberband_baseline_correction(wavelengths,spectra_col_3)

        # path_4 = r'../res/rat/kidney_oct/'
        # col_data_4 = loadmat(os.path.join(path_4, "HMT_5_after_mask1.mat"))
        # spectra_col_4 = np.reshape(col_data_4['spectrum'], (426))
        # baseline_1,spectra_col_als_4 = rubberband_baseline_correction(wavelengths,spectra_col_4)
        # plt.figure(figsize=(10, 7))

        # Assume wavelengths and rubberband_baseline_correction are already defined
        foldername_list = ['8020SER','8020PSER','7030SER','7030PSER','6040SER','6040PSER','1000']
        sub_folder_list = {'LMT_1','LMT_2'} #'LMT_1',
        for fl in foldername_list:
            for sub in sub_folder_list:
                folders = [
                    f'C:/pyws/SPEC/res/Caf2_10302025/{fl}/{sub}',
                    # r'C:/pyws/SPEC/res/Caf2_09242025_amide1_2nd/9010SER/LMT_3'
                    # r'D:/spec_res/rat/Caf2_03132025_rat_ffpe/liver_ffpe/HMT_5_mask2_corrected_spectra',
                    # r'D:/spec_res/rat/kidney_ffpe/HMT_1',
                    # r'D:/spec_res/rat/liver_ff/HMT_1',
                    # r'D:/spec_res/rat/kidney_ff/HMT_1',
                ]
                # save_path = folders
                # labels = ['liver_ffpe', 'kidney_ffpe', 'liver_ff', 'kidney_ff']
                # colors = ['orange', 'blue', 'green', 'purple']
                labels = ['raw']
                colors = ['orange']

                plt.figure(figsize=(10, 7))

                for folder, label, color in zip(folders, labels, colors):
                    spectra_all = []
                    for i, file in enumerate(os.listdir(folder)):
                        if file.endswith('.npy'):
                            # print(f"{folder} - {i}")
                            spectrum = np.load(os.path.join(folder, file))
                            _, corrected = rubberband_baseline_correction(wavelengths, spectrum)
                            plt.plot(wavelengths, corrected)
                            spectra_all.append(corrected)

                    spectra_all = np.array(spectra_all)
                    mean_spectrum = np.mean(spectra_all, axis=0)
                    std_spectrum = np.std(spectra_all, axis=0)

                    # plt.plot(wavelengths, mean_spectrum, label=label, color=color)
                    # plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                    #                 color=color, alpha=0.2)

                plt.xlabel("Wavelength (cm⁻¹)", fontsize=14)
                plt.ylabel("Normalized Intensity (a.u.)", fontsize=14)
                plt.title("Average ± STD Spectra", fontsize=16)
                # plt.legend(loc='upper left', fontsize=12)
                plt.grid(True)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(f'C:/pyws/SPEC/res/Caf2_10302025/{fl}/{sub}/spec_vis.png'))
                # plt.show()

        # # Colored background for each biomolecule region
        # plt.axvspan(1720, 1755, color='purple', alpha=0.1, label='Lipids')
        # plt.axvspan(1610, 1690, color='green', alpha=0.1, label='Proteins')
        # plt.axvspan(1500, 1600, color='green', alpha=0.1)
        # plt.axvspan(1215, 1245, color='blue', alpha=0.1, label='Nucleic Acids')
        # plt.axvspan(1065, 1095, color='blue', alpha=0.1)
        # plt.axvspan(1020, 1120, color='orange', alpha=0.1, label='Carbohydrates')
        # plt.plot(wavelengths, spectra_col_als_1, label='liver_ffpe',color='orange',linewidth=3) 
        # plt.plot(wavelengths, spectra_col_als_2, label='kidney_ffpe',color='blue',linewidth=3) 
        # plt.plot(wavelengths, spectra_col_als_3, label='liver_ff',color='green',linewidth=3) 
        # plt.plot(wavelengths, spectra_col_als_4, label='kidney_ff',color='purple',linewidth=3) 
        # plt.xlabel('Wavenumber (cm⁻¹)', fontsize=14)
        # plt.ylabel('Intensity', fontsize=14)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.legend(loc='upper left', fontsize=14)
        # plt.title(f"Absorption Spectrum")
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