import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from pdb import set_trace as st

def process_spectrum(spectrum, wavelengths, start, end, window=13, polyorder=2, prominence=0.0003):
        indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
        second_derivative = savgol_filter(spectrum, window_length=window, polyorder=polyorder, deriv=2)
        minima_indices, _ = find_peaks(-second_derivative, prominence=prominence)
        minima_x = wavelengths[indices][minima_indices]
        minima_y = second_derivative[minima_indices]
        return second_derivative, minima_x, minima_y

def amideI_normalize(wn, y, lo=1600, hi=1700):
    wn = np.asarray(wn).ravel()
    y = np.asarray(y).ravel()

    mask = (wn >= lo) & (wn <= hi)
    if not np.any(mask):
        raise ValueError(f"No points in Amide I window {lo}-{hi} cm^-1. Check wavenumber array.")
    peak = np.max(y[mask])
    if peak == 0:
        raise ValueError("Amide I peak is 0, cannot normalize.")
    return y / peak, peak

def main():
    foldername_list = ['/'] #['kidney_oct/','liver_oct/','kidney_ffpe/','liver_ffpe/']
    for foldername in foldername_list:
        input_folder = r'../res/03232026_col1+4/CAF2/org/mean_spec/'+f'{foldername}'
        output_folder = r'../res/03232026_col1+4/CAF2/org/second_derivative/'+f'{foldername}'
        os.makedirs(output_folder, exist_ok=True)
        csv_file = os.path.join(output_folder, "detected_peaks.csv")

        # Wavelength range (cm⁻¹)
        wavelength_start = 950
        wavelength_end = 1800
        wavelengths = np.linspace(950, 1800, 426)  # Assuming this range for all subspectra
        # Initialize CSV data
        csv_data = []

        # Iterate through all .mat files in the input folder
        for file in os.listdir(input_folder):
            mat_path = os.path.join(input_folder, file)
            if file.endswith(".npy"):
                data = np.load(mat_path)
                spectra = data
            elif file.endswith(".mat"):
                data = loadmat(mat_path)
                spectra = data['spectrum'].flatten()
            else:
                continue
            # spectra, col1_peak = amideI_normalize(wavelengths, spectra, lo=1600, hi=1700)
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
            # plt.figure(figsize=(10, 2))
            # plt.plot(wavelengths, second_derivative,label="Second Derivative", color="k", linewidth=2)
            # plt.scatter(minima_x, minima_y, color="red", label="Local Minima")
            # for x, y in zip(minima_x, minima_y):
            #     plt.annotate(f'{x:.0f}', xy=(x, y), xytext=(x, y + 0.00003),
            #                 fontsize=14, ha='center', arrowprops=dict(arrowstyle='->', color='blue', lw=0.5))
            # # plt.xlabel("Wavenumber (cm⁻¹)")
            # plt.ylabel("Second Derivative")
            # plt.legend(loc="upper left")
            # plt.axis('off')
            # # plt.title(f"Second Derivative of {file}")
            # # Save the plot
            # plot_filename = f"2nd_{file.replace('.mat', '').replace('.npy', '')}.png"
            # # plt.show()
            # # st()
            # plt.savefig(os.path.join(output_folder, plot_filename))
            # plt.close()
            # print(f"Done processing {file}.")
            threshold = -0.00005  # tune this

            filtered = [(x, y) for x, y in zip(minima_x, minima_y) if y < threshold]
            minima_x_plot = [x for x, y in filtered]
            minima_y_plot = [y for x, y in filtered]

            plt.figure(figsize=(10, 2))
            plt.plot(wavelengths, second_derivative, color="k", linewidth=2)
            plt.scatter(minima_x_plot, minima_y_plot, color="red", s=20)

            for x, y in zip(minima_x_plot, minima_y_plot):
                plt.annotate(
                    f"{x:.0f}",
                    xy=(x, y),
                    xytext=(x, y - 0.0015),
                    fontsize=14,
                    ha="center"
                    # arrowprops=dict(arrowstyle="", color="blue", lw=0.5)
                )

            plt.axis("off")
            # plt.margins(x=0.01, y=0.15)
            # plt.ylabel("Second Derivative")
            # plt.legend(loc="upper left")
            plot_filename = f"2nd_{file.replace('.mat', '').replace('.npy', '')}.png"
            plt.savefig(
                os.path.join(output_folder, plot_filename),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
                transparent=True
            )
            plt.close()

            print(f"Done processing {file}.")
    
        # Save detected peaks to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"Processed all spectra. Results saved to {csv_file}")


if __name__ == '__main__':
    main()