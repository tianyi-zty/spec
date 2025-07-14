import os
import numpy as np
import random
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from pdb import set_trace as st

def process_spectrum(spectrum, wavelengths, start, end, window=13, polyorder=2, prominence=0.0003):
    indices = np.where((wavelengths >= start) & (wavelengths <= end))[0]
    second_derivative = savgol_filter(spectrum, window_length=window, polyorder=polyorder, deriv=2)
    minima_indices, _ = find_peaks(second_derivative[indices], prominence=prominence)
    minima_x = wavelengths[indices][minima_indices]
    minima_y = second_derivative[indices][minima_indices]
    return second_derivative, minima_x, minima_y

def main():
    #change foldername also root path
    foldername = 'afterCollagen60Peptide40'
    filename_before = 'LMR_1'
    filename_after = 'LMR_1'
    region = 1
    print(f'Processing before:{filename_before}/after:{filename_after} at amide {region} region')
    root_path = '/Volumes/TIANYI/Sperodata/06122025_AUPILLAR_ETCHED_MEM/6'
    before_path = os.path.join(root_path, 'before', f'{filename_before}.mat')
    after_path = os.path.join(root_path, foldername, f'{filename_after}.mat')
    save_path = f'/Volumes/TIANYI/spec_res/06122025_AUPILLAR_ETCHED_MEM/{region}/{foldername}/{filename_after}/'
    second_deriv_dir = os.path.join(save_path, "second_derivative_spec")
    save_path_filtered_spec = os.path.join(save_path, "filtered_spec")
    os.makedirs(second_deriv_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_filtered_spec, exist_ok=True)

    # Load spectral cubes
    spectra_before = np.reshape(loadmat(before_path)['r'], (480, 480, 426))
    spectra_after = np.reshape(loadmat(after_path)['r'], (480, 480, 426))
    wavelengths = np.linspace(950, 1800, 426)

    # Define wavelength subset indices (1400–1700 cm⁻¹)
    subset_start = 1400
    subset_end = 1700
    subset_indices = np.where((wavelengths >= subset_start) & (wavelengths <= subset_end))[0]

    # ROI selection
    a=88
    b=52
    c=64
    d=82
    x_start, x_end = a,a+100
    y_start, y_end = b,b+100
    x_start_1, x_end_1 = c,c+100
    y_start_1, y_end_1 = d,d+100
    region_before = spectra_before[x_start:x_end, y_start:y_end, :]
    region_after = spectra_after[x_start_1:x_end_1, y_start_1:y_end_1, :]

    # Save ROI inspection image
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.imshow(spectra_before[:, :, 330].T)
    ax1.set_title('spectra_before')
    ax2.imshow(spectra_after[:, :, 330].T)
    ax2.set_title('spectra_after')
    ax3.imshow(region_before[:, :, 330].T)
    ax3.set_title('extracted_region_before')
    ax4.imshow(region_after[:, :, 330].T)
    ax4.set_title('extracted_region_after')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cropped_image_example.png'))
    # plt.show()
    plt.close()
    # st()

    # Apply Otsu threshold to 330th band of region_after
    thresholds = threshold_multiotsu(region_after[:, :, 330].T, classes=3)
    regions = np.digitize(region_after[:, :, 330].T, bins=thresholds)
    binary_mask_region0 = (regions == 0).astype(np.uint8)
    binary_mask_region1 = (regions == 1).astype(np.uint8)
    binary_mask_region2 = (regions == 2).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    ax1, ax2 = axes.flatten()
    ax1.imshow(region_after[:, :, 330].T)
    ax1.set_title('region_after')
    ax2.imshow(binary_mask_region2)
    ax2.set_title('binary_mask_region2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'binary_mask_example.png'))
    # plt.show()
    plt.close()
    # st()

    data = binary_mask_region2.T[:, :, np.newaxis]*region_after
    # combined_mask = (binary_mask_region1 + binary_mask_region2).T[:, :, np.newaxis]
    # data = combined_mask * region_after
    # st()
    # Collect valid spectra from masked, smoothed region
    valid_spectra = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            spectrum = data[i, j, :]
            # Apply Gaussian smoothing (3x3 spatial only)
            # spectrum = gaussian_filter(spectrum, sigma=(3, 3, 0))
            if np.any(spectrum):
                valid_spectra.append(spectrum)

    valid_spectra = np.array(valid_spectra)
    # st()
    # Compute and plot mean ± std spectrum
    mean_spectrum = np.mean(valid_spectra, axis=0)
    std_spectrum = np.std(valid_spectra, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, mean_spectrum, label='Mean Spectrum', color='black', linewidth=3)
    plt.fill_between(wavelengths,
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        color='gray', alpha=0.4, label='±1 Std Dev')
    plt.title(f"Mean ± Std Spectrum ({foldername} / {filename_after})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mean_std_spectrum.png'))
    plt.show()
    plt.close()
    # st()

    # Now filter based on second derivative peaks between 1600–1650
    selected_spectra = []
    for idx, spectrum in enumerate(valid_spectra):
        second_derivative, minima_x, minima_y = process_spectrum(
            spectrum, wavelengths, 1500, 1650, prominence=0.0003
        )
        if len(minima_x) > 2:
            selected_spectra.append(spectrum)

            # # Plotting the second derivative with peaks
            # plt.figure(figsize=(8, 3))
            # plt.plot(wavelengths, second_derivative, label="2nd Derivative", color="black")
            # plt.scatter(minima_x, minima_y, color="red", label="Peaks")
            # for x, y in zip(minima_x, minima_y):
            #     plt.annotate(f'{x:.0f}', xy=(x, y), xytext=(x, y + 0.0002),
            #                 fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', color='blue', lw=0.5))
            # plt.title(f"2nd Derivative Spectrum #{idx}")
            # plt.xlabel("Wavenumber (cm⁻¹)")
            # plt.ylabel("2nd Derivative")
            # plt.tight_layout()
            # plt.savefig(os.path.join(second_deriv_dir, f"second_derivative_{idx}.png"))
            # plt.close()

    print(f"Found {len(selected_spectra)} spectra with peak between 1500–1650 cm⁻¹.")
    print(f"Saved plots in: {second_deriv_dir}")

    # Randomly save up to 2000 spectra
    random.seed(42)
    n_samples = min(len(selected_spectra), 2000)
    sampled = random.sample(selected_spectra, n_samples)
    # for idx, spectrum in enumerate(sampled):
    #     np.save(os.path.join(save_path_filtered_spec, f"spectrum_{idx}.npy"), spectrum)
    for idx, spectrum in enumerate(sampled):
        spectrum_subset = spectrum[subset_indices]  # Slice the spectrum
        np.save(os.path.join(save_path_filtered_spec, f"spectrum_{idx}.npy"), spectrum_subset.astype(np.float32))

        # Compute and slice second derivative
        second_derivative, _, _ = process_spectrum(spectrum, wavelengths, subset_start, subset_end)
        second_derivative_subset = second_derivative[subset_indices]
        np.save(os.path.join(second_deriv_dir, f"second_derivative_{idx}.npy"), second_derivative_subset.astype(np.float32))
    print(f"✅ Saved {n_samples} original spectra to {save_path_filtered_spec}")
    print(f"✅ Saved {n_samples} 2nd derivative spectra to {second_deriv_dir}")

if __name__ == '__main__':
    main()
