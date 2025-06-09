import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st




def main():
        
        ####plot spectrum for single chip at different resonance
        # chipname = 'AuPillars_50nmAl2O3_1_05222025'
        # save_path = f'../res/after_cleaning/peptide1/{chipname}/figures/'
        # os.makedirs(save_path, exist_ok=True)
        
        # filename = [1,2,3,4,5,6,7,8]
        # for fn in filename:
        #     print('filename:',fn)
        #     # Plot the average spectrum with standard deviation
        #     plt.figure(figsize=(12, 8))
        #     path = f'../res/after_cleaning/peptide1/{chipname}/{fn}/'
            
        #     for file in os.listdir(path):
        #         if file.endswith(".mat"):
        #             mat_path = os.path.join(path, file)
        #             # st()
        #             name = file.split('.mat')[0]
        #             wavelengths = np.linspace(950, 1800, 426)
        #             data_after = loadmat(mat_path)
        #             spectra_after = np.reshape(data_after['spectrum'], (426))
        #             plt.plot(wavelengths, spectra_after, label=f'{name}',linewidth=2) 
        #             plt.xlabel('Wavenumber (cm⁻¹)')
        #             plt.ylabel('Intensity')
        #             plt.ylim((0,1.2))
        #             plt.legend(loc='upper left')
        #             plt.title(f"Spectrum {filename}")

        #     # plt.show()
        #     plt.savefig(os.path.join(save_path, f'spectrum visualization_{fn}.png'))
        #     # Save

        ####plot spectrum for single resonance for different chips(dif concentration)
        chipname = ['AuPilllars_10nmAl2O3_Cleaning_05122025','AuPillars_50nmAl2O3_1_05222025','AuPillars_50nmAl2O3_2_05222025','AuPillars_50nmAl2O3_3_05222025','AuPillars_50nmAl2O3_4_05232025','AuPillars_50nmAl2O3_6_05232025']
        save_path = f'../res/after_cleaning/peptide1/figures/'
        os.makedirs(save_path, exist_ok=True)
        
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
            plt.savefig(os.path.join(save_path, f'spectrum visualization_{filename}.png'))
            # Save

if __name__ == '__main__':
    main()