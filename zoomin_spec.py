__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import os
from scipy.spatial import ConvexHull
import numpy as np


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

    # path = r'../data/AuPillars_10212024/beforeCollagen'
    collagen_path = r'../res/rat/Caf2_03072025_rat_oct/kidney_oct'
    collagen_path_1 = r'../res/rat/Caf2_03072025_rat_oct/liver_oct'
    collagen_path_2 = r'../res/rat/Caf2_03132025_rat_ffpe/kidney_ffpe'
    collagen_path_3 = r'../res/rat/Caf2_03132025_rat_ffpe/liver_ffpe'
    save_path = '../res/rat/'
    os.makedirs(save_path, exist_ok=True)

    sorted_mat = np.sort(glob(os.path.join(collagen_path, '*.mat')))
    sorted_mat_1 = np.sort(glob(os.path.join(collagen_path_1, '*.mat')))
    sorted_mat_2 = np.sort(glob(os.path.join(collagen_path_2, '*.mat')))
    sorted_mat_3 = np.sort(glob(os.path.join(collagen_path_3, '*.mat')))
    len_ = sorted_mat.shape[0]
    len_1 = sorted_mat_1.shape[0]
    len_2 = sorted_mat_2.shape[0]
    len_3 = sorted_mat_3.shape[0]

    wavenumbers = np.linspace(950, 1800, 426)

    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = {
    'z_index0' : np.where((wavenumbers>=950)&(wavenumbers<=1800))
    # 'z_index1' : np.where((wavenumbers>=1450)&(wavenumbers<=1460)),
    # 'z_index2': np.where((wavenumbers>=1536)&(wavenumbers<=1546)),
    # 'z_index3' : np.where((wavenumbers>=1634)&(wavenumbers<=1644)),
    # 'z_index4' : np.where((wavenumbers>=1650)&(wavenumbers<=1660)),
    # 'z_index5' : np.where((wavenumbers>=1660)&(wavenumbers<=1670))
    # 'z_index6' : np.where((wavenumbers>=1660)&(wavenumbers<=1680))
    }

    data_list= []
    for ind, f in enumerate(sorted_mat):
        print(f'{ind+1}/{len_}')
        # st()
        # name = f.split('\\')[-1]
        # print('processing file:',name)
        data = loadmat(f)
        data = data['spectrum'].flatten()
        data_list.append(data)
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum = np.mean(np.array(data_list), axis=0)
    baseline_before, corrected_spectrum_before = rubberband_baseline_correction(wavenumbers, mean_spectrum)
    std_spectrum = np.std(np.array(data_list), axis=0)+0.03

    data_list_1= []
    for ind, f in enumerate(sorted_mat_1):
        print(f'{ind+1}/{len_1}')
        # st()
        # name = f.split('\\')[-1]
        # print('processing file:',name)
        data = loadmat(f)
        data = data['spectrum'].flatten()
        data_list_1.append(data)
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_1 = np.mean(np.array(data_list_1), axis=0)
    baseline_before, corrected_spectrum_before_1 = rubberband_baseline_correction(wavenumbers, mean_spectrum_1)
    std_spectrum_1 = np.std(np.array(data_list_1), axis=0)+0.035

    data_list_2= []
    for ind, f in enumerate(sorted_mat_2):
        print(f'{ind+1}/{len_2}')
        # st()
        # name = f.split('\\')[-1]
        # print('processing file:',name)
        data = loadmat(f)
        data = data['spectrum'].flatten()
        data_list_2.append(data)
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_2 = np.mean(np.array(data_list_2), axis=0)
    baseline_before, corrected_spectrum_before_2 = rubberband_baseline_correction(wavenumbers, mean_spectrum_2)
    std_spectrum_2 = np.std(np.array(data_list_2), axis=0)+0.04

    data_list_3= []
    for ind, f in enumerate(sorted_mat_3):
        print(f'{ind+1}/{len_3}')
        # st()
        # name = f.split('\\')[-1]
        # print('processing file:',name)
        data = loadmat(f)
        data = data['spectrum'].flatten()
        data_list_3.append(data)
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_3 = np.mean(np.array(data_list_3), axis=0)
    baseline_before, corrected_spectrum_before_3 = rubberband_baseline_correction(wavenumbers, mean_spectrum_3)
    std_spectrum_3 = np.std(np.array(data_list_3), axis=0)+0.038

    for i in range(0,1):
        idx = wavelengths[f'z_index{i}'][0]

        # Plot the average spectrum with standard deviation
        plt.figure(figsize=(8, 6))
        # Plot the mean spectrum as a solid line
        # plt.plot(wavenumbers[idx], corrected_spectrum_before_3[idx], label='liver_ffpe', color='#ff7f0e', linewidth=2)
        # plt.plot(wavenumbers[idx], corrected_spectrum_before_2[idx], label='kidney_ffpe', color='#1f77b4', linewidth=2)
        # plt.plot(wavenumbers[idx], corrected_spectrum_before_1[idx]-0.02, label='liver_oct', color='#2ca02c', linewidth=2)
        plt.plot(wavenumbers[idx], corrected_spectrum_before[idx], label='kidney_oct', color='b', linewidth=2)

        plt.fill_between(wavenumbers[idx],
                        corrected_spectrum_before[idx] - std_spectrum_1[idx],
                        corrected_spectrum_before[idx] + std_spectrum_1[idx],
                        color='b', alpha=0.1)
        # plt.fill_between(wavenumbers[idx],
        #                 corrected_spectrum_before_1[idx]-0.02 - std_spectrum_1[idx],
        #                 corrected_spectrum_before_1[idx]-0.02 + std_spectrum_1[idx],
        #                 color='#2ca02c', alpha=0.1)
        # plt.fill_between(wavenumbers[idx],
        #                 corrected_spectrum_before_2[idx] - std_spectrum_3[idx],
        #                 corrected_spectrum_before_2[idx] + std_spectrum_3[idx],
        #                 color='#1f77b4', alpha=0.1)
        # plt.fill_between(wavenumbers[idx],
        #                 corrected_spectrum_before_3[idx] - std_spectrum_3[idx],
        #                 corrected_spectrum_before_3[idx] + std_spectrum_3[idx],
        #                 color='#ff7f0e', alpha=0.1)

        # plt.xlabel('Wavenumber (cm⁻¹)')
        # plt.ylabel('Intensity')
        plt.ylim(0, 0.5)
        # plt.title(f'Average Spectrum and Standard Deviation: z_index{i}')
        plt.legend(loc='upper left', fontsize=12)
        # plt.grid(True)

        plt.savefig(os.path.join(save_path, f'average_spectrum_region_1000_1100.png'))
        print(f"Saved: average_spectrum_region_1000_1100.png")
        plt.show()
        plt.close()
        # st()

    # # Plot the average spectrum with standard deviation
    # plt.figure(figsize=(12, 8))
    # # Plot the mean spectrum as a solid line
    # plt.plot(wavelengths[z_index1], mean_spectrum[z_index1], label='1000_Average Spectrum', color='b', linewidth=2)
    # plt.plot(wavelengths[z_index1], mean_spectrum_1[z_index1], label='8020_Average Spectrum', color='r', linewidth=2)
    # # Fill the area representing standard deviation
    # plt.fill_between(wavelengths[z_index1], 
    #                 mean_spectrum[z_index1] - std_spectrum[z_index1], 
    #                 mean_spectrum[z_index1] + std_spectrum[z_index1], 
    #                 color='b', alpha=0.3, label='1000_Standard Deviation')
    # plt.fill_between(wavelengths[z_index1], 
    #                 mean_spectrum_1[z_index1] - std_spectrum_1[z_index1], 
    #                 mean_spectrum_1[z_index1] + std_spectrum_1[z_index1], 
    #                 color='r', alpha=0.3, label='8020_Standard Deviation')
    # # Labeling the plot
    # plt.xlabel('Wavenumber (cm⁻¹)')
    # plt.ylabel('Intensity')
    # plt.title(f'Average Spectrum and Standard Deviation')
    # plt.legend()
    # plt.grid(True)
    # # plt.show()
    # # Save
    # plt.savefig(os.path.join(save_path, f'average_spectrum_region_{np.min(z_index1)}-{np.max(z_index1)}.png'))
    # plt.close()


    # # Plot the average spectrum with standard deviation
    # plt.figure(figsize=(12, 8))
    # # Plot the mean spectrum as a solid line
    # plt.plot(wavelengths[z_index2], mean_spectrum[z_index2], label='1000_Average Spectrum', color='b', linewidth=2)
    # plt.plot(wavelengths[z_index2], mean_spectrum_1[z_index2], label='8020_Average Spectrum', color='r', linewidth=2)
    # # Fill the area representing standard deviation
    # plt.fill_between(wavelengths[z_index2], 
    #                 mean_spectrum[z_index2] - std_spectrum[z_index2], 
    #                 mean_spectrum[z_index2] + std_spectrum[z_index2], 
    #                 color='b', alpha=0.3, label='1000_Standard Deviation')
    # plt.fill_between(wavelengths[z_index2], 
    #                 mean_spectrum_1[z_index2] - std_spectrum_1[z_index2], 
    #                 mean_spectrum_1[z_index2] + std_spectrum_1[z_index2], 
    #                 color='r', alpha=0.3, label='8020_Standard Deviation')
    # # Labeling the plot
    # plt.xlabel('Wavenumber (cm⁻¹)')
    # plt.ylabel('Intensity')
    # plt.title(f'Average Spectrum and Standard Deviation')
    # plt.legend()
    # plt.grid(True)
    # # plt.show()
    # # Save
    # plt.savefig(os.path.join(save_path, f'average_spectrum_region_{np.min(z_index2)}-{np.max(z_index2)}.png'))    
    # plt.close()



    # # Plot the average spectrum with standard deviation
    # plt.figure(figsize=(12, 8))
    # # Plot the mean spectrum as a solid line
    # plt.plot(wavelengths[z_index3], mean_spectrum[z_index3], label='1000_Average Spectrum', color='b', linewidth=2)
    # plt.plot(wavelengths[z_index3], mean_spectrum_1[z_index3], label='8020_Average Spectrum', color='r', linewidth=2)
    # # Fill the area representing standard deviation
    # plt.fill_between(wavelengths[z_index3], 
    #                 mean_spectrum[z_index3] - std_spectrum[z_index3], 
    #                 mean_spectrum[z_index3] + std_spectrum[z_index3], 
    #                 color='b', alpha=0.3, label='1000_Standard Deviation')
    # plt.fill_between(wavelengths[z_index3], 
    #                 mean_spectrum_1[z_index3] - std_spectrum_1[z_index3], 
    #                 mean_spectrum_1[z_index3] + std_spectrum_1[z_index3], 
    #                 color='r', alpha=0.3, label='8020_Standard Deviation')
    # # Labeling the plot
    # plt.xlabel('Wavenumber (cm⁻¹)')
    # plt.ylabel('Intensity')
    # plt.title(f'Average Spectrum and Standard Deviation')
    # plt.legend()
    # plt.grid(True)
    # # plt.show()
    # # Save
    # plt.savefig(os.path.join(save_path, f'average_spectrum_region_{np.min(z_index3)}-{np.max(z_index3)}.png'))
    # plt.close()


if __name__ == '__main__':
    main()