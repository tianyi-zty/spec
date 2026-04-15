import os
from scipy.io import loadmat
from scipy.signal import correlate
from scipy.io import savemat
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as st


def process_block(block_after, wavelengths, save_path, block_id):
    """
    Process a 10x10 block of spectra, save the results.
    """
    # Calculate the mean spectrum for the block
    mean_spectrum_after = np.mean(block_after, axis=(0, 1))

    # Save the subspectrum
    subspectrum_filename = f"{block_id}.mat"
    savemat(os.path.join(save_path+'/spectrum', subspectrum_filename), {'spectrum': mean_spectrum_after})
    print(f'Subspectrum for {block_id} saved!')


    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, mean_spectrum_after, label='Spectrum (caf2)', color='r')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc='upper left')
    plt.title(f"Spectrum for {block_id}")
    plt.savefig(os.path.join(save_path+'/figures/', f"{block_id}_Spectrum.png"))
    plt.close()

def main():
    foldername_list = ['1000']  # '1000','9010','8020','7030'
    filename_list = ['LMT_1']
    # filename_list = ['LMT_1','LMT_2','LMT_3'] #'HMT_3','HMT_4','HMT_5','HMT_6','HMT_7','HMT_8','HMT_9','HMT_10'
    for filename in filename_list:
        after_collagen = f'/Volumes/TIANYI/Sperodata/Caf2_09022025/1000/{filename}.mat'
        save_path = f'../res/Caf2_09092025_tsnefilter/1000/{filename}/'
        # os.makedirs(save_path, exist_ok=True)
        # os.makedirs(save_path+'/spectrum', exist_ok=True)
        # os.makedirs(save_path+'/figures', exist_ok=True)

        wavelengths = np.linspace(950, 1800, 426)
        data_after = loadmat(after_collagen)
        spectra_after = np.reshape(data_after['r'], (480, 480, 426))
        # spectra_after = spectra_after/np.max(spectra_after)
        # spectra_after = (spectra_after - np.min(spectra_after)) / (np.max(spectra_after) - np.min(spectra_after))

        # st()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(spectra_after[:,:,330])
        ax.set_title('spectra')
        plt.tight_layout()
        plt.show()
        st()
        
        plt.savefig(os.path.join(save_path+'/figures', 'spectra image visualization'+f'{filename}'+'.png'))
        plt.close()
        st()

        mean_spectrum = np.mean(spectra_after, axis=(0, 1))
        std_spectrum = np.std(spectra_after, axis=(0, 1))

        # Plot the average spectrum with standard deviation
        plt.figure(figsize=(10, 8))
        # Plot the mean spectrum as a solid line
        plt.plot(wavelengths, mean_spectrum, label='Average Spectrum', color='b', linewidth=2) 
        # Fill the area representing standard deviation
        plt.fill_between(wavelengths, 
                        mean_spectrum - std_spectrum, 
                        mean_spectrum + std_spectrum, 
                        color='b', alpha=0.3, label='Standard Deviation')
        # Labeling the plot
        plt.xlabel('Wavenumber (cm⁻¹)',fontsize=18)
        plt.ylabel('Intensity (a.u.)',fontsize=18)
        # plt.title(f'Average Spectrum and Standard Deviation {filename}')
        plt.legend(fontsize=14)
        # plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.show()
        # Save
        plt.savefig(os.path.join(save_path, f'average_spectrum_{filename}.png'))
        plt.close()
        print(f"Spectrum saved.")
        # st()


        block_size = 480

        for i in range(0, spectra_after.shape[0], block_size):
            for j in range(0, spectra_after.shape[1], block_size):
                block_after = spectra_after[i:i+block_size, j:j+block_size, :]
                # st()
                if block_after.shape[0] == block_size and block_after.shape[1] == block_size:
                    process_block(block_after, wavelengths, save_path, filename)
                # st()


if __name__ == '__main__':
    main()
