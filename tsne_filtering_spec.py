import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.io import loadmat
from pdb import set_trace as st
import random

def normalize_spectra_zscore(X):
    """Apply z-score normalization per spectrum (row-wise)."""
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    return (X - X_mean) / (X_std + 1e-8)

def elbow_method(X, max_k=10, save_path=None):
    """Run elbow method to find optimal k."""
    inertias = []
    K = range(1, max_k + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        print(f"k={k}, inertia={inertias[-1]:.2f}")

    # Plot inertia vs k
    plt.figure(figsize=(6, 4))
    plt.plot(K, inertias, 'bo-', linewidth=2, markersize=6)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Elbow plot saved to {save_path}")
    else:
        plt.show()

    return inertias

def main():
    foldername_list = ['kidney_oct']  # '1000','9010', '8020','7030'
    filename_list = ['HMT_1']  # 'LMT_1','LMT_2','LMT_3','LMT_4'

    for foldername in foldername_list:
        for filename in filename_list:
            print('processing:', foldername, filename)
            after_collagen = f'/Volumes/TIANYI/Sperodata/Caf2_03072025_rat_oct/{foldername}/{filename}.mat'
            save_path = f'../res/rat/{foldername}/{filename}/'
            os.makedirs(save_path, exist_ok=True)

            # Load data
            data_after = loadmat(after_collagen)
            spectra_after = np.reshape(data_after['r'], (480, 480, 426))
            wavelengths = np.linspace(950, 1800, 426)
            # Crop region
            # region_after = spectra_after[0:480, 0:480, :]
            data = spectra_after

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(data[:,:,330])
            ax.set_title('spectra')
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(save_path, 'spectra image visualization'+f'{filename}'+'.png'))
            plt.close()
            st()


            # Collect valid spectra
            valid_spectra = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    spectrum = data[i, j, :]
                    if not np.any(spectrum):
                        continue
                    valid_spectra.append(spectrum)

            valid_spectra = np.array(valid_spectra)
            print(f"Found {valid_spectra.shape[0]} spectra in {foldername}/{filename}")

            # Normalize (optional: comment/uncomment as needed)
            # valid_spectra_norm = normalize_spectra_zscore(valid_spectra)
            valid_spectra_norm = valid_spectra

            # PCA
            pca_n = min(50, valid_spectra_norm.shape[0], valid_spectra_norm.shape[1])
            pca = PCA(n_components=pca_n, random_state=42)
            X_pca = pca.fit_transform(valid_spectra_norm)

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
            X_tsne = tsne.fit_transform(X_pca)

            # Run Elbow method
            elbow_method(X_tsne, max_k=5,
                         save_path=os.path.join(save_path, "elbow_plot.png"))
            # st()
            
if __name__ == "__main__":
    main()
