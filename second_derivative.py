__author__ = 'Tianyi'

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import os
from glob import glob

def clean_axes(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')

def process_spectrum(file_path, wavelengths, wavelength_start=950, wavelength_end=1800):
    """
    Load .npy spectrum, calculate second derivative, find minima, and return data for plotting.
    """
    # Load the .npy file
    spectrum = np.load(file_path).flatten()

    # Select wavelength range
    z_indices = np.where((wavelengths >= wavelength_start) & (wavelengths <= wavelength_end))[0]
    selected_wavelengths = wavelengths[z_indices]
    selected_spectrum = spectrum[z_indices]

    # Second derivative using Savitzky-Golay filter
    second_derivative = savgol_filter(selected_spectrum, window_length=13, polyorder=2, deriv=2)

    # Find local minima (peaks in inverted second derivative)
    minima_indices, _ = find_peaks(-second_derivative, prominence=0.0003)
    minima_x = selected_wavelengths[minima_indices]
    minima_y = second_derivative[minima_indices]

    return selected_wavelengths, second_derivative, minima_x, minima_y

def main():
    # Folder containing .npy files
    folder_path = 'C:/pyws/SPEC/res/Caf2_09302025/mean_spec/'
    save_path = 'C:/pyws/SPEC/res/Caf2_09302025/2ndplots/'
    os.makedirs(save_path, exist_ok=True)

    # Wavelength range (cm⁻¹)
    wavelengths = np.linspace(950, 1800, 426)  # adjust to match your original data

    # Find all .npy files in folder
    files = glob(os.path.join(folder_path, '*.npy'))
    if not files:
        print("No .npy files found in the folder!")
        return


    # Process each file
    for file_path in files:
        plt.figure(figsize=(10, 2))
        # Use the filename (without extension) as label
        label = os.path.splitext(os.path.basename(file_path))[0]

        wl, sec_deriv, minima_x, minima_y = process_spectrum(file_path, wavelengths)

        plt.plot(wl, sec_deriv, label=f'Second Derivative: {label}', linewidth=2, color='black')
        plt.scatter(minima_x, minima_y, color='red', marker='o', label=f'Local Minima: {label}')

        # Annotate minima
        # for x, y in zip(minima_x, minima_y):
        #     plt.annotate(f'({x:.0f}, {y:.4f})',
        #                 xy=(x, y),
        #                 xytext=(x, y),
        #                 fontsize=18,
        #                 ha='center',
        #                 arrowprops=dict(arrowstyle='->', color='red', lw=1))

        print(f'{label} detected minima:', minima_x)
        clean_axes()
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        # plt.legend(loc='upper right')
        plt.title('Second Derivative')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{label}_Second_Derivative.png'))
        # plt.show()

if __name__ == '__main__':
    main()
