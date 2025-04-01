import os
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from pdb import set_trace as st


def process_pixel(spectra, wavelengths, save_path, x, y, pixel_id):
    """
    Extract and save the spectrum for a specific (x, y) pixel.
    """
    pixel_spectrum = spectra[x, y, :]  # Extract the spectrum

    # Ensure save directories exist
    subspectrum_dir = os.path.join(save_path, 'subspectrum')
    figures_dir = os.path.join(save_path, 'figures')
    os.makedirs(subspectrum_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save spectrum as .mat file
    subspectrum_filename = os.path.join(subspectrum_dir, f"pixel_{pixel_id}_({x},{y}).mat")
    savemat(subspectrum_filename, {'spectrum': pixel_spectrum})
    print(f'Spectrum for pixel ({x}, {y}) saved at {subspectrum_filename}')

    # Plot and save the spectrum
    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, pixel_spectrum, label=f'Pixel ({x},{y}) Spectrum', color='b')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend(loc='upper left')
    plt.title(f"Spectrum for Pixel ({x}, {y})")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    figure_filename = os.path.join(figures_dir, f"Pixel_{pixel_id}_({x},{y}).png")
    plt.savefig(figure_filename)
    plt.close()

def main():
    filename = 'HMR_1_4D'
    input_file = rf'W:/3. Students/Tianyi/Agcube/03272025_spincoating/{filename}.mat'
    save_path = f'../res/Agcube/03272025_spincoating/{filename}'

    os.makedirs(save_path, exist_ok=True)

    wavelengths = np.linspace(950, 1800, 426)  # Define the wavelength range

    # Load .mat file
    try:
        data = loadmat(input_file)
        spectra = np.reshape(data['r'], (480, 480, 426))  # Ensure correct shape
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(spectra[:,:,350])
    ax.set_title('spectra')
    plt.tight_layout()
    plt.show()
    st()
    # plt.savefig(os.path.join(save_path, 'spectra image visualization.png'))
    # st()


    # x_start, x_end = 240, 270 
    # y_start, y_end = 260, 290 
    # region_after = spectra[x_start:x_end, y_start:y_end, :]
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.imshow(region_after[:,:,350])
    # ax.set_title('spectra')
    # plt.tight_layout()
    # plt.show()
    # st()
    # plt.savefig(os.path.join(save_path, 'spectra image visualization.png'))
    # st()


    # Define the list of (x, y) coordinates to analyze
    pixel_coordinates = [(277,167),(282,93),(302,123)]  # Update this list as needed (211,253),(222,258),(234,263),(254,276),(283,263)

    # Process each selected pixel
    for pixel_id, (x, y) in enumerate(pixel_coordinates):
        if 0 <= x < spectra.shape[0] and 0 <= y < spectra.shape[1]:  # Ensure valid indices
            process_pixel(spectra, wavelengths, save_path, x, y, pixel_id)
        else:
            print(f"Skipping invalid coordinate ({x}, {y})")

if __name__ == '__main__':
    main()
