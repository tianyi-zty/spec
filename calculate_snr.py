import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

def load_tif_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Could not read image: {path}")
    return image.astype(np.float32)

def find_signal_roi(image, size=20):
    y, x = np.unravel_index(np.argmax(image), image.shape)
    half = size // 2
    y1, y2 = max(0, y - half), min(image.shape[0], y + half)
    x1, x2 = max(0, x - half), min(image.shape[1], x + half)
    return image[y1:y2, x1:x2]

def find_background_roi(image, size=20):
    # Select top-left corner as background (can customize this logic)
    return image[0:size, 0:size]

def calculate_contrast(signal_region):
    I_max = np.max(signal_region)
    I_min = np.min(signal_region)
    return (I_max - I_min) / (I_max + I_min + 1e-10)

def calculate_snr(signal_region, background_region):
    signal = np.mean(signal_region)
    noise = np.std(background_region)
    return signal / (noise + 1e-10)

def calculate_strehl_ratio(image, fwhm_pixels, peak_intensity=None):
    if peak_intensity is None:
        peak_intensity = np.max(image)

    sigma = fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))
    theoretical_psf = np.zeros_like(image)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    theoretical_psf[center] = 1
    theoretical_psf = gaussian_filter(theoretical_psf, sigma=sigma)
    theoretical_peak = np.max(theoretical_psf)

    return peak_intensity / (theoretical_peak + 1e-10)

# --- Settings ---
folder_path = r"/Users/tianyizheng/Desktop/postdoc/myproject/shgondoublehole/canbeused/processed/2channelrotatecrop/forward/"
output_csv = "results.csv"
fwhm_pixels = 1
roi_size = 20

# --- Process all images ---
results = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".tif", ".tiff")):
        file_path = os.path.join(folder_path, filename)
        try:
            image = load_tif_image(file_path)
            signal_region = find_signal_roi(image, size=roi_size)
            background_region = find_background_roi(image, size=roi_size)

            contrast = calculate_contrast(signal_region)
            snr = calculate_snr(signal_region, background_region)
            strehl = calculate_strehl_ratio(image, fwhm_pixels)

            results.append({
                "Filename": filename,
                "Contrast": contrast,
                "SNR": snr,
                "Strehl Ratio": strehl
            })

            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# --- Save Results ---
df = pd.DataFrame(results)
df.to_csv(folder_path+output_csv, index=False)
print(f"\nResults saved to {output_csv}")
