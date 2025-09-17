import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pdb import set_trace as st
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.cluster import KMeans
import umap
from collections import defaultdict

# ---------------------- #
# Data Loading Functions #
# ---------------------- #
def load_npy_data(folder, max_samples=1000):
    print(f"Processing folder: {folder}")
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    if len(files) == 0:
        print(f"⚠️ Nothing in this folder: {folder}")
        return np.array([])
    np.random.seed(42)
    files = np.random.choice(files, size=min(max_samples, len(files)), replace=False)
    data = []
    for file in files:
        arr = np.load(os.path.join(folder, file))
        if np.any(arr):
            data.append(arr.flatten())
    return np.array(data)

def normalize_spectra_zscore(X):
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    return (X - X_mean) / (X_std + 1e-8)

# -------------------------- #
# Plotting Function: Avg/STD #
# -------------------------- #
def plot_avg_std_with_top_features(X, y, labels, top_feature_indices, top_importance, save_path, title, zoom_ranges=None):
    wavelengths = np.linspace(950, 1800, X.shape[1])
    unique_labels = np.unique(y)
    cmap = plt.get_cmap('tab20')

    # --- Full Spectrum Plot ---
    plt.figure(figsize=(12, 8))
    for label in unique_labels:
        group_data = X[y == label]
        avg = np.mean(group_data, axis=0)
        std = np.std(group_data, axis=0)
        plt.plot(wavelengths, avg, label=labels[label], color=cmap(label), linewidth=2.5)
        plt.fill_between(wavelengths, avg - std, avg + std, color=cmap(label), alpha=0.2)

    for idx in top_feature_indices:
        plt.plot([wavelengths[idx],wavelengths[idx]], [-2,-1.5], color='black', linestyle='-', linewidth=1.5)

    plt.title(f"{title} (Full Spectrum)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_')}_full_spectrum.png"))
    plt.show()
    plt.close()

    # --- Zoomed-in Regions ---
    if zoom_ranges:
        for i, (start, end) in enumerate(zoom_ranges):
            idx = np.where((wavelengths >= start) & (wavelengths <= end))[0]
            plt.figure(figsize=(10, 6))
            for label in unique_labels:
                group_data = X[y == label]
                avg = np.mean(group_data, axis=0)
                std = np.std(group_data, axis=0)
                plt.plot(wavelengths[idx], avg[idx], label=labels[label], color=cmap(label), linewidth=2.5)
                plt.fill_between(wavelengths[idx], avg[idx] - std[idx], avg[idx] + std[idx], color=cmap(label), alpha=0.2)

            # for idx_feat in top_feature_indices:
            #     if start <= wavelengths[idx_feat] <= end:
            #         plt.axvline(wavelengths[idx_feat], color='black', linestyle='-', linewidth=1.5)

            plt.title(f"{title} Zoom: {start}-{end} cm⁻¹")
            plt.xlabel("Wavenumber (cm$^{-1}$)")
            plt.ylabel("Normalized Intensity")
            # plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_')}_zoom_{start}_{end}.png"))
            plt.show()
            plt.close()

# ---------------------- #
# Main Analysis Pipeline #
# ---------------------- #
def main():
    foldername_list = ['1000', '9010', '8020', '7030', '6040'] #'1000B','1000', , '9010', '8020', '7030', '6040'
    filename_list = ['LMT_1'] # 'LMT_2','LMT_3',
    
    base_path = "D:/spec_res/Caf2_07182025_amide1/"
    save_path = "D:/spec_res/Caf2_07182025_amide1/result/"
    os.makedirs(save_path, exist_ok=True)

    all_data = []
    all_labels = []
    label_names = []
    label_index = 0

    for foldername in foldername_list:
        for filename in filename_list:
            folder_path = os.path.join(base_path, foldername, filename)
            if not os.path.isdir(folder_path):
                continue
            data = load_npy_data(folder_path, max_samples=500)
            if len(data) == 0:
                continue
            norm_data = normalize_spectra_zscore(data)
            all_data.append(norm_data)
            all_labels += [label_index] * len(norm_data)
            label_names.append(f"{foldername}_{filename}")
            print(f"Loaded {len(norm_data)} data from {foldername}/{filename} as label {label_index}")
            label_index += 1

    X = np.concatenate(all_data)
    y = np.array(all_labels)
    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:1000]
    y_small = y_small[:1000]

    # # -------------------- #
    # # PCA + t-SNE + UMAP  #
    # # -------------------- #
    # X_pca = PCA(n_components=50).fit_transform(X_small)
    # X_tsne = TSNE(n_components=2, perplexity=200, random_state=42).fit_transform(X_pca)
    # X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_pca)

    # --------------------------- #
    # Feature Importance (LogReg) #
    # --------------------------- #
    clf = LogisticRegression(penalty='l2', solver='liblinear')
    clf.fit(X_small, y_small)
    importance = np.mean(np.abs(clf.coef_), axis=0)
    wavenumbers = np.linspace(950, 1800, len(importance))
    top_indices = np.argsort(importance)[-80:]
    top_wavenumbers = wavenumbers[top_indices]
    top_importance = importance[top_indices]

    plt.figure(figsize=(24, 15))
    plt.plot(wavenumbers, importance, color='darkred', linewidth=1.5)
    plt.title("Feature Importance by Wavenumber (Logistic Regression)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Importance (|Weight|)")
    plt.grid(True)
    for x, y_val in zip(top_wavenumbers, top_importance):
        plt.annotate(f'{x:.0f}', xy=(x, y_val), xytext=(x, y_val + 0.01),
                     fontsize=10, ha='center',
                     arrowprops=dict(arrowstyle='->', color='blue', lw=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Feature_Importance.png'))
    plt.close()

    # --------------------- #
    # Dimensionality Plots #
    # --------------------- #
    def plot_embedding(embedding, name, coords_label):
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab20')
        for i, label in enumerate(np.unique(y_small)):
            plt.scatter(embedding[y_small == label, 0], embedding[y_small == label, 1],
                        label=label_names[label], alpha=0.5, s=30, color=cmap(i))
        plt.title(f"{name} Projection")
        plt.xlabel(f"{coords_label} 1")
        plt.ylabel(f"{coords_label} 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{name}_zscore.png'))
        plt.show()
        plt.close()

    # plot_embedding(X_tsne, 't-SNE', 't-SNE')
    # plot_embedding(X_umap, 'UMAP', 'UMAP')

    # # ------------------------------ #
    # # Confusion Matrix + Clustering #
    # # ------------------------------ #
    # n_clusters = len(np.unique(y_small))
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_tsne)
    # tsne_labels = kmeans.labels_
    # conf_mat = confusion_matrix(y_small, tsne_labels)
    # row_ind, col_ind = linear_sum_assignment(-conf_mat)
    # label_map = {col: row for row, col in zip(col_ind, row_ind)}
    # mapped_tsne_labels = np.array([label_map[lbl] for lbl in tsne_labels])

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=[f"Cluster {i}" for i in range(n_clusters)],
    #             yticklabels=[label_names[i] for i in range(n_clusters)])
    # plt.xlabel("KMeans Cluster Label")
    # plt.ylabel("True Label")
    # plt.title("Confusion Matrix (t-SNE KMeans vs. True Labels)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, "Confusion_Matrix_tsne.png"))
    # plt.close()

    # ------------------------ #
    # Avg ± STD Spectrum Plot #
    # ------------------------ #
    zoom_ranges = [
        (1210,1264),
        (1536,1570),
        (1630,1700)
        # (1520, 1570),
        # (1640, 1700),
        # (1730, 1740)
    ]
    plot_avg_std_with_top_features(X_small, y_small, label_names, top_indices, top_importance,
                                   save_path, title="Avg STD Spectrum with Top Features",
                                   zoom_ranges=zoom_ranges)

if __name__ == '__main__':
    main()
