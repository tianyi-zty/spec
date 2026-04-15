import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

# ---- Config ----
folders = ["liver_ff", "liver_ffpe"]
subfolders = [f"HMT_{i}" for i in range(1, 2)]
data_dir = r'/Volumes/TIANYI/spec_res/rat/'
file_ext = "1.npy"   # adjust if needed

def load_spectrum_file(filepath):
    """Load a single spectrum file."""
    if filepath.endswith(".npy"):
        return np.load(filepath)
    elif filepath.endswith(".mat"):
        mat = loadmat(filepath)
        return np.array(mat['spectrum']).squeeze()  # adjust key if needed
    elif filepath.endswith(".csv"):
        return np.loadtxt(filepath, delimiter=",")
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

def average_spectrum_from_folder(folder_path):
    spectra = []
    for f in os.listdir(folder_path):
        if f.endswith(file_ext):
            spectrum = load_spectrum_file(os.path.join(folder_path, f))
            spectra.append(spectrum)
    if len(spectra) == 0:
        return None
    return np.mean(np.vstack(spectra), axis=0)

# ---- Load data ----
avg_spectra = {}
labels = []

for folder in folders:
    for sub in subfolders:
        folder_path = os.path.join(data_dir, folder, sub)
        if os.path.isdir(folder_path):
            avg = average_spectrum_from_folder(folder_path)
            if avg is not None:
                key = f"{folder}_ROIs"
                avg_spectra[key] = avg
                labels.append(key)

# ---- Compute correlation ----
stacked = np.vstack([avg_spectra[k] for k in labels])
corr_matrix = np.corrcoef(stacked)

# ---- Plot heatmap ----
plt.figure(figsize=(5, 4))
sns.heatmap(
    corr_matrix,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    annot_kws={"size": 24}, 
    cmap="coolwarm",
    vmin=0.93, vmax=1
)
# plt.title("Correlation Matrix (Kidney FF vs FFPE)", fontsize=16)
plt.title("Average spectrum (Liver FF vs FFPE)", fontsize=16)
plt.tight_layout()
plt.show()
