import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from pdb import set_trace as st


def main():
    # Paths
    input_folder = r"../res/AuPillars_Al2O3_12102024/1/pixel/95-5/subspectra"
    output_folder = r"../res/AuPillars_Al2O3_12102024/1/pixel/95-5/2ndplot"
    csv_file = os.path.join(output_folder, "detected_peaks.csv")

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Wavelength range (cm⁻¹)
    wavelength_start = 1400
    wavelength_end = 1700
    wavelengths = np.linspace(950, 1800, 426)  # Assuming this range for all subspectra
    # Initialize CSV data
    csv_data = []

    def process_spectrum(spectrum, wavelengths, start, end, window=13, polyorder=2, prominence=0.001):
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
                ############## need to ynamically adjust the wavelengths array based on the spectrum size################
                spectrum_size = len(spectra)
                wavelengths = np.linspace(1400, 1700, spectrum_size)
                # Process the spectrum
                second_derivative, minima_x, minima_y = process_spectrum(
                    spectra, wavelengths, wavelength_start, wavelength_end
                )
                # st()
                # Save detected peaks to CSV data
                csv_data.append([file, *minima_x])
                # st()
                # # Plot the second derivative with detected peaks
                # plt.figure(figsize=(10, 6))
                # plt.plot(wavelengths, second_derivative, label="Second Derivative", color="k", linewidth=2)
                # plt.scatter(minima_x, minima_y, color="red", label="Local Minima")
                # for x, y in zip(minima_x, minima_y):
                #     plt.annotate(f'({x:.0f}, {y:.4f})', xy=(x, y), xytext=(x, y + 0.0001),
                #                 fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='blue', lw=0.5))
                # plt.xlabel("Wavenumber (cm⁻¹)")
                # plt.ylabel("Intensity")
                # plt.legend(loc="upper right")
                # plt.title(f"Second Derivative of {file}")
                
                # # Save the plot
                # plot_filename = f"Second_Derivative_with_coordinates_{file.replace('.mat', '')}.png"
                # plt.show()
                # st()
                # plt.savefig(os.path.join(output_folder, plot_filename))
                # plt.close()
                print(f"Done processing {file}.")
            
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Save detected peaks to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print(f"Processed all spectra. Results saved to {csv_file}")


if __name__ == '__main__':
    main()