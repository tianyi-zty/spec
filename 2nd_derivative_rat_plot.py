import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks

def process_spectrum(spectrum, wavelengths, start, end, window=13, polyorder=2, prominence=0.0001):
    second_derivative = savgol_filter(spectrum, window_length=window, polyorder=polyorder, deriv=2)
    indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
    minima_indices, _ = find_peaks(-second_derivative, prominence=prominence)
    minima_x = wavelengths[minima_indices]
    minima_y = second_derivative[minima_indices]
    return second_derivative, minima_x, minima_y

def main():
    foldername_list = ['kidney_oct/', 'liver_oct/', 'kidney_ffpe/', 'liver_ffpe/']
    color_map = {
        'kidney_oct/': '#9467bd',
        'liver_oct/': '#2ca02c',
        'kidney_ffpe/': '#1f77b4',
        'liver_ffpe/': '#ff7f0e'
    }

    plt.figure(figsize=(4, 6))

    for foldername in foldername_list:
        input_folder = os.path.join('/Volumes/TIANYI/rat/', foldername)
        files = [f for f in os.listdir(input_folder) if f.endswith('1_after_mask1.mat')]
        
        for file in files:
            mat_path = os.path.join(input_folder, file)
            try:
                data = loadmat(mat_path)
                spectra = data['spectrum'].flatten()
                spectrum_size = len(spectra)
                wavelengths = np.linspace(950, 1800, spectrum_size)
                second_derivative, minima_x, minima_y = process_spectrum(
                    spectra, wavelengths, 950, 1800
                )
                
                # Plot zoomed region only
                zoom_indices = np.where((wavelengths >= 1000) & (wavelengths <= 1060))[0]
                plt.plot(wavelengths[zoom_indices],
                         second_derivative[zoom_indices],
                         label=foldername.strip('/'),
                         color=color_map[foldername],
                         linewidth=3)
                break  # Only plot one spectrum per group

            except Exception as e:
                print(f"Error processing {file} in {foldername}: {e}")

    # plt.xlabel("Wavenumber (cm⁻¹)", fontsize=14)
    # plt.ylabel("Second Derivative", fontsize=14)
    # plt.title("Zoomed Second Derivative (1000–1060 cm⁻¹)", fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    
    # # Add arrows at specific wavenumbers
    # arrow_positions = [1026, 1030, 1046]
    # ymin, ymax = plt.gca().get_ylim()
    # for x in arrow_positions:
    #     plt.annotate(
    #         f'{x}', 
    #         xy=(x, ymin + 0.0001),        # Arrowhead
    #         xytext=(x, ymin),    # Start just below axis
    #         ha='center',
    #         va='top',
    #         fontsize=12,
    #         arrowprops=dict(arrowstyle='->', color='black', lw=2)
    #     )

    plt.tight_layout()
    save_path = '../res/rat/zoomin_spec/combined_zoomed_plot.png'
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved combined plot to {save_path}")

if __name__ == '__main__':
    main()
