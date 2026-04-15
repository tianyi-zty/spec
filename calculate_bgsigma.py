import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os



def compute_second_derivative_sigma(spectrum_path, wavelengths=None, range_min=950, range_max=1800):
    """
    Compute the sigma (standard deviation) of the smoothed 2nd derivative
    of a single spectrum in a specified wavelength range.

    Parameters:
        spectrum_path (str): Path to the .npy spectrum file.
        wavelengths (np.array): 1D array of wavelength values (cm^-1). If None, defaults to 950-1800 (426 points).
        range_min (float): Start of wavelength range for sigma calculation.
        range_max (float): End of wavelength range for sigma calculation.

    Returns:
        sigma (float): Standard deviation of 2nd derivative in the specified range.
        second_derivative (np.array): Smoothed 2nd derivative of the spectrum.
        smoothed_spectrum (np.array): Smoothed spectrum.
    """
    # Load spectrum
    spectrum = np.load(spectrum_path)  # should be shape (426,)

    if wavelengths is None:
        wavelengths = np.linspace(950, 1800, len(spectrum))

    # Smooth spectrum
    smoothed_spectrum = savgol_filter(spectrum, window_length=11, polyorder=3)

    # Compute 2nd derivative
    second_derivative = savgol_filter(smoothed_spectrum, window_length=11, polyorder=3, deriv=2)

    # Select range for sigma
    range_mask = (wavelengths >= range_min) & (wavelengths <= range_max)
    sigma = np.std(second_derivative[range_mask])

    return sigma, second_derivative, smoothed_spectrum, wavelengths

def plot_spectrum_and_derivative(spectrum_path, range_min=950, range_max=1800):
    sigma, second_derivative, smoothed_spectrum, wavelengths = compute_second_derivative_sigma(
        spectrum_path, range_min=range_min, range_max=range_max
    )

    # Plot original spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, smoothed_spectrum, color='blue', label='Smoothed Spectrum')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Smoothed Spectrum')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("/Volumes/TIANYI/res/rat_otsu/liver_ff_bg.png"), dpi=300)
    plt.show()

    # Plot 2nd derivative with ±sigma lines
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, second_derivative, color='red', label='2nd Derivative')
    plt.axhline(sigma, color='gray', linestyle='--', label=f'+σ = {sigma:.4f}')
    plt.axhline(-sigma, color='gray', linestyle='--', label=f'-σ = {-sigma:.4f}')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('2nd Derivative (a.u.)')
    plt.title('2nd Derivative Spectrum with σ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("/Volumes/TIANYI/res/rat_otsu/liver_ff_bg_sigma.png"), dpi=300)
    plt.show()

    print(f"Sigma of 2nd derivative in [{range_min}, {range_max}]: {sigma:.4f}")

if __name__ == "__main__":
    spectrum_file = "/Volumes/TIANYI/res/rat_otsu/liver_ff_bg.npy"  
    plot_spectrum_and_derivative(spectrum_file, range_min=950, range_max=1800)
