import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st




def main():
        # Plot the average spectrum with standard deviation
        plt.figure(figsize=(12, 8))

        path = r'../res/Agcube/04292025_0.1mgml_colgel_ployonsubstrate/HMR_4B_1/subspectrum/'
        filename = '04292025_0.1mgml_colgel_ployonsubstrate/HMR_4B_1'

        for file in os.listdir(path):
            if file.endswith(".mat"):
                mat_path = os.path.join(path, file)
                # st()
                name = file.split('.mat')[0]
                wavelengths = np.linspace(950, 1800, 426)
                data_after = loadmat(mat_path)
                spectra_after = np.reshape(data_after['spectrum'], (426))
                plt.plot(wavelengths, spectra_after, label=f'{name}',linewidth=2) 
                plt.xlabel('Wavenumber (cm⁻¹)')
                plt.ylabel('Intensity')
                plt.legend(loc='upper left')
                plt.title(f"Spectrum {filename}")

        # plt.show()
        plt.savefig(os.path.join(path, 'spectrum visualization.png'))
            # Save


if __name__ == '__main__':
    main()