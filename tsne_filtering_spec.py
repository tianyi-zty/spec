import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.io import loadmat
# ---------------------- #
# Data Loading Functions #
# ---------------------- #

def load_npy_data(folder):
    """Load .npy files from a folder and randomly sample up to `max_samples`."""
    print(f"Processing folder: {folder}")
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    if len(files) == 0:
        print(f"⚠️ Nothing in this folder: {folder}")
        return np.array([])  # Return empty array
    
    np.random.seed(42)
    files = np.random.choice(files, replace=False)

    data = []
    for file in files:
        arr = np.load(os.path.join(folder, file))
        if np.any(arr):  # skip all-zero arrays
            data.append(arr.flatten())
    return np.array(data)

def normalize_spectra_zscore(X):
    """Apply z-score normalization per spectrum."""
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    return (X - X_mean) / (X_std + 1e-8)

# ---------------------- #
# Main Analysis Pipeline #
# ---------------------- #

def main():
    foldername_list = ['1000'] # '1000','9010', '8020','7030' 
    filename_list = ['LMT_1'] #'LMT_1','LMT_2','LMT_3','LMT_4'
    for foldername in foldername_list:
        for filename in filename_list:
            print('processing:', foldername, filename)
            # before_collagen = r'/Volumes/TIANYI/Sperodata/CaF2_01162025/after_sample/LMT_1.mat'
            after_collagen = f'W:/3. Students/Tianyi/Caf2_09092025/{foldername}/{filename}'+'.mat'
            save_path = f'../res/Caf2_09092025_amide1/{foldername}/{filename}/'
            os.makedirs(save_path, exist_ok=True)
            # save_2nd = f'../res/Caf2_09092025_amide1/{foldername}/{filename}/'
            # os.makedirs(save_2nd, exist_ok=True)

            data_after = loadmat(after_collagen)
            spectra_after = np.reshape(data_after['r'], (480, 480, 426))
            wavelengths = np.linspace(950, 1800, 426)  # Assuming this range for all subspectra
            wavelength_start = 950
            wavelength_end = 1800

            x_start, x_end = 0,480
            y_start, y_end = 0,480
            region_after = spectra_after[x_start:x_end, y_start:y_end, :]
    base_path = "D:/spec_res/rat/liver_ffpe/HMT_1"  
    save_path = "D:/spec_res/rat/result_single/"
    os.makedirs(save_path, exist_ok=True)

    # Load data
    X = load_npy_data(base_path)
    if X.size == 0:
        return

    # Normalize
    X = normalize_spectra_zscore(X)

    # PCA (for denoising & speeding up t-SNE)
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X)

    # t-SNE
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)

    # KMeans clustering on t-SNE
    n_clusters = 4   # 👈 set number of clusters you want
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_tsne)
    labels = kmeans.labels_

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("t-SNE Clustering of Spectra")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "tsne_clustering.png"))
    plt.show()

if __name__ == "__main__":
    main()
