__author__ = 'Tianyi'

import spectrochempy as scp
from scipy.io import loadmat
from pdb import set_trace as st
from matplotlib import pyplot as plt
from glob import glob
import os
import numpy as np




def main():

    # path = r'../data/AuPillars_10212024/beforeCollagen'
    collagen_path = r'W:/3. Students/TianYi/Caf2_11152024/aftercollagen'
    save_path = '../res/Caf2_11152024'

    mat_list = glob(os.path.join(collagen_path,'LMT_1.mat'))
    # Sort the files based on the extracted numbers
    sorted_mat = np.sort(mat_list)
    # st()
    len_ = sorted_mat.shape[0]

    # Wavelength range (cm⁻¹)
    wavelength_start = 950
    wavelength_end = 1800
    # Assume the wavelength step size (426 points between 950 and 1800)
    wavelengths = np.linspace(950, 1800, 426)

    for ind, f in enumerate(sorted_mat):
        print(f'{ind+1}/{len_}')
        # st()
        name = f.split('\\')[-1]
        print('processing file:',name)

        # Load the .mat file
        data = loadmat(f)
        # st()
        spectra = np.reshape(data['r'], (480, 480, 426))

        st()
        # Define the region of interest on the x and y axes
        x_start, x_end = 100, 400 #200, 400  #100, 200 #  # Replace with your desired x range
        y_start, y_end = 100, 400  # Replace with your desired y range

        # Find the indices corresponding to the desired wavelength range
        z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]

        # Extract the subregion and wavelength range
        extracted_region = spectra[x_start:x_end, y_start:y_end, z_indices]

        ###################### Determine the global intensity range for consistent scaling #######################
        global_min = min(spectra.min(), extracted_region.min())
        global_max = max(spectra.max(), extracted_region.max())

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(spectra[:,:,0], cmap='gray', vmin=global_min, vmax=global_max)
        plt.subplot(1, 2, 2)
        plt.imshow(extracted_region[:,:,0], cmap='gray', vmin=global_min, vmax=global_max)

        plt.tight_layout()
        plt.show()
        # Save
        # plt.savefig(os.path.join(save_path, f'cropped_image_w0_{name}_{x_start}_{x_end}_{y_start}_{y_end}.png'))
        print(f"Cropped image saved.")
        # st()
        ######################

        ####################### Get the shape of the extracted region #######################
        # num_x, num_y, num_z = extracted_region.shape

        # # Create a figure for plotting all spectra
        # plt.figure(figsize=(12, 8))

        # # Loop over all pixels in the extracted region and plot the spectrum
        # for i in range(num_x):
        #     for j in range(num_y):
        #         pixel_spectrum = extracted_region[i, j, :]
                
        #         # Plot the spectrum for the current pixel
        #         plt.plot(wavelengths[z_indices], pixel_spectrum, alpha=0.5, linewidth=0.8)

        # # Labeling the plot
        # plt.xlabel('Wavenumber (cm⁻¹)')
        # plt.ylabel('Intensity')
        # plt.title(f'Spectra of All Pixels in Region ({x_start}:{x_end}, {y_start}:{y_end})')
        # plt.grid(True)
        # plt.show()
        # st()
        #######################

        # Calculate the mean and standard deviation across all pixels
        mean_spectrum = np.mean(extracted_region, axis=(0, 1))
        std_spectrum = np.std(extracted_region, axis=(0, 1))

        # Plot the average spectrum with standard deviation
        plt.figure(figsize=(12, 8))
        
        # Plot the mean spectrum as a solid line
        plt.plot(wavelengths[z_indices], mean_spectrum, label='Average Spectrum', color='b', linewidth=2)
        
        # Fill the area representing standard deviation
        plt.fill_between(wavelengths[z_indices], 
                        mean_spectrum - std_spectrum, 
                        mean_spectrum + std_spectrum, 
                        color='b', alpha=0.3, label='Standard Deviation')

        # Labeling the plot
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title(f'Average Spectrum and Standard Deviation in Region ({x_start}:{x_end}, {y_start}:{y_end})')
        plt.legend()
        plt.grid(True)
        plt.show()
        # Save
        # plt.savefig(os.path.join(save_path, f'average_spectrum_region_{name}_{x_start}_{x_end}_{y_start}_{y_end}.png'))
        print(f"Spectrum saved.")
        # st()

        


if __name__ == '__main__':
    main()