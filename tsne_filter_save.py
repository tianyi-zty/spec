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


def main():
    foldername_list = ['kidney_oct']  # '1000','9010','8020','7030'
    filename_list = ['HMT_3','HMT_4','HMT_5']   # 'LMT_1','LMT_2','LMT_3','LMT_4'

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

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(spectra_after[:,:,330])
            ax.set_title('spectra')
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(save_path, 'spectra image visualization '+f'{filename}'+'.png'))
            plt.close()
            # st()

            # Flatten image -> spectra list
            h, w, d = spectra_after.shape
            reshaped = spectra_after.reshape(-1, d)

            # Mask valid spectra (non-zero)
            mask = np.any(reshaped != 0, axis=1)
            valid_spectra = reshaped[mask]
            valid_indices = np.where(mask)[0]  # keep mapping

            print(f"Found {valid_spectra.shape[0]} spectra in {foldername}/{filename}")

            # Normalize
            # valid_spectra_norm = normalize_spectra_zscore(valid_spectra)
            valid_spectra_norm = valid_spectra

            # PCA
            pca_n = min(50, valid_spectra_norm.shape[0], valid_spectra_norm.shape[1])
            pca = PCA(n_components=pca_n, random_state=42)
            X_pca = pca.fit_transform(valid_spectra_norm)

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
            X_tsne = tsne.fit_transform(X_pca)

            # KMeans with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_tsne)

            # Save cluster labels
            np.save(os.path.join(save_path, "tsne_clusters_2.npy"), labels)

            # ==========================
            # Plot clustering map (480x480)
            # ==========================
            cluster_map = np.full(h * w, fill_value=-1)
            cluster_map[valid_indices] = labels
            cluster_map = cluster_map.reshape(h, w)

            plt.figure(figsize=(6, 6))
            plt.imshow(cluster_map, cmap="viridis")
            plt.title(f"Cluster Map ({foldername}/{filename})")
            plt.colorbar(label="Cluster ID")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "cluster_map.png"), dpi=300)
            plt.close()

            # ==========================
            # Save 10,000 spectra from each cluster
            # ==========================
            for cluster_id in range(2):
                cluster_indices = np.where(labels == cluster_id)[0]
                n_to_save = min(10000, len(cluster_indices))
                chosen_idx = random.sample(list(cluster_indices), n_to_save)

                cluster_folder = os.path.join(save_path, f"cluster_{cluster_id}")
                os.makedirs(cluster_folder, exist_ok=True)

                np.save(os.path.join(cluster_folder, "spectra.npy"), valid_spectra[chosen_idx])
                np.save(os.path.join(cluster_folder, "indices.npy"), chosen_idx)

                print(f"Saved {n_to_save} spectra for cluster {cluster_id}")

            print(f"Finished {foldername}/{filename}")
            st()


if __name__ == "__main__":
    main()
