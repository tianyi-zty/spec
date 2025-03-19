import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from pdb import set_trace as st


def main():
    input_folder = r"../res/Caf2_03072025_rat/liver_ffpe/HMT_6/subspectrum"
    output_folder = r"../res/Caf2_03072025_rat/liver_ffpe/HMT_6/result"
    os.makedirs(output_folder, exist_ok=True)
    csv_file = os.path.join(output_folder, "detected_peaks.csv")

    # Wavelength range (cm⁻¹)
    wavelength_start = 950
    wavelength_end = 1800
    wavelengths = np.linspace(950, 1800, 426)  # Assuming this range for all subspectra
    # Initialize CSV data
    csv_data = []

    def process_spectrum(spectrum, wavelengths, start, end, window=13, polyorder=2, prominence=0.0006):
        indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
        second_derivative = savgol_filter(spectrum, window_length=window, polyorder=polyorder, deriv=2)
        minima_indices, _ = find_peaks(-second_derivative, prominence=prominence)
        minima_x = wavelengths[indices][minima_indices]
        minima_y = second_derivative[minima_indices]
        return second_derivative, minima_x, minima_y

    # Iterate through all .mat files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".mat"):
            mat_path = os.path.join(input_folder, file)
            try:
                # Load spectrum data
                data = loadmat(mat_path)
                # st()
                spectra = data['spectrum'].flatten()
                # st()
                ############## need to dnamically adjust the wavelengths array based on the spectrum size################
                spectrum_size = len(spectra)
                wavelengths = np.linspace(wavelength_start, wavelength_end, spectrum_size)
                # Process the spectrum
                second_derivative, minima_x, minima_y = process_spectrum(
                    spectra, wavelengths, wavelength_start, wavelength_end
                )
                # st()
                # Save detected peaks to CSV data
                csv_data.append([file, *minima_x])
                # st()
                # Plot the second derivative with detected peaks
                plt.figure(figsize=(10, 6))
                plt.plot(wavelengths, second_derivative, label="Second Derivative", color="k", linewidth=2)
                plt.scatter(minima_x, minima_y, color="red", label="Local Minima")
                for x, y in zip(minima_x, minima_y):
                    plt.annotate(f'({x:.0f}, {y:.4f})', xy=(x, y), xytext=(x, y + 0.0001),
                                fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='blue', lw=0.5))
                plt.xlabel("Wavenumber (cm⁻¹)")
                plt.ylabel("Intensity")
                plt.legend(loc="upper right")
                plt.title(f"Second Derivative of {file}")
                # Save the plot
                plot_filename = f"Second_Derivative_with_coordinates_{file.replace('.mat', '')}.png"
                # plt.show()
                # st()
                plt.savefig(os.path.join(output_folder, plot_filename))
                plt.close()
                print(f"Done processing {file}.")
            
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Save detected peaks to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print(f"Processed all spectra. Results saved to {csv_file}")


if __name__ == '__main__':
    main()