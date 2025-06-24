__author__ = 'Tianyi'

from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import os
import numpy as np




def main():

    # path = r'../data/AuPillars_10212024/beforeCollagen'
    collagen_path = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/1000'
    collagen_path_1 = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/8020'
    collagen_path_2 = r'../res/caf2_06132025/tnse/result/3group/original/clustered_data/6040'
    save_path = '../res/caf2_06132025/tnse/result/3group/original/'

    sorted_mat = np.sort(glob(os.path.join(collagen_path, '*.npy')))
    sorted_mat_1 = np.sort(glob(os.path.join(collagen_path_1, '*.npy')))
    sorted_mat_2 = np.sort(glob(os.path.join(collagen_path_2, '*.npy')))
    len_ = sorted_mat.shape[0]
    len_1 = sorted_mat_1.shape[0]
    len_2 = sorted_mat_2.shape[0]
    wavenumbers = np.linspace(950, 1800, 426)

    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = {
    'z_index0' : np.where((wavenumbers>=950)&(wavenumbers<=960)),
    'z_index1' : np.where((wavenumbers>=1450)&(wavenumbers<=1460)),
    'z_index2': np.where((wavenumbers>=1536)&(wavenumbers<=1546)),
    'z_index3' : np.where((wavenumbers>=1634)&(wavenumbers<=1644)),
    'z_index4' : np.where((wavenumbers>=1650)&(wavenumbers<=1660)),
    'z_index5' : np.where((wavenumbers>=1660)&(wavenumbers<=1670))
    # 'z_index6' : np.where((wavenumbers>=1660)&(wavenumbers<=1680))
    }

    data_list= []
    for ind, f in enumerate(sorted_mat):
        print(f'{ind+1}/{len_}')
        # st()
        # name = f.split('\\')[-1]
        # print('processing file:',name)
        data = np.load(f)
        data_list.append(data)
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum = np.mean(np.array(data_list), axis=0)
    std_spectrum = np.std(np.array(data_list), axis=0)

    data_list_1= []
    for ind, f in enumerate(sorted_mat_1):
        print(f'{ind+1}/{len_1}')
        # st()
        # name = f.split('\\')[-1]
        # print('processing file:',name)
        data = np.load(f)
        data_list_1.append(data)
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_1 = np.mean(np.array(data_list_1), axis=0)
    std_spectrum_1 = np.std(np.array(data_list_1), axis=0)

    data_list_2= []
    for ind, f in enumerate(sorted_mat_2):
        print(f'{ind+1}/{len_2}')
        # st()
        # name = f.split('\\')[-1]
        # print('processing file:',name)
        data = np.load(f)
        data_list_2.append(data)
    # st()
    # Calculate the mean and standard deviation across all pixels
    mean_spectrum_2 = np.mean(np.array(data_list_2), axis=0)
    std_spectrum_2 = np.std(np.array(data_list_2), axis=0)

    for i in range(0,6):
        idx = wavelengths[f'z_index{i}'][0]

        # Plot the average spectrum with standard deviation
        plt.figure(figsize=(12, 8))
        # Plot the mean spectrum as a solid line
        plt.plot(wavenumbers[idx], mean_spectrum[idx], label='1000_Average Spectrum', color='b', linewidth=2)
        plt.plot(wavenumbers[idx], mean_spectrum_1[idx], label='8020_Average Spectrum', color='r', linewidth=2)
        plt.plot(wavenumbers[idx], mean_spectrum_2[idx], label='6040_Average Spectrum', color='g', linewidth=2)

        plt.fill_between(wavenumbers[idx],
                        mean_spectrum[idx] - std_spectrum[idx],
                        mean_spectrum[idx] + std_spectrum[idx],
                        color='b', alpha=0.3, label='1000_Standard Deviation')
        plt.fill_between(wavenumbers[idx],
                        mean_spectrum_1[idx] - std_spectrum_1[idx],
                        mean_spectrum_1[idx] + std_spectrum_1[idx],
                        color='r', alpha=0.3, label='8020_Standard Deviation')
        plt.fill_between(wavenumbers[idx],
                        mean_spectrum_2[idx] - std_spectrum_2[idx],
                        mean_spectrum_2[idx] + std_spectrum_2[idx],
                        color='g', alpha=0.3, label='6040_Standard Deviation')
        
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title(f'Average Spectrum and Standard Deviation: z_index{i}')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(save_path, f'average_spectrum_region_z_index{i}.png'))
        print(f"Saved: average_spectrum_region_z_index{i}.png")
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