import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
import umap
from pdb import set_trace as st

# --- Normalization Functions ---
def normalize_spectra_zscore(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)

# --- Data Loader ---
def load_npy_data(folder, max_samples=2000):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    np.random.seed(42)
    files = np.random.choice(files, size=min(max_samples, len(files)), replace=False)
    data = [np.load(os.path.join(folder, f)).flatten() for f in files if np.any(np.load(os.path.join(folder, f)))]
    return data

# --- Plot Feature Importance ---
def plot_feature_importance(importance, wavenumbers, save_path):
    top_indices = np.argsort(importance)[-20:]
    top_wavenumbers = wavenumbers[top_indices]
    top_importance = importance[top_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(wavenumbers, importance, color='darkred', linewidth=1.5)
    plt.title("Feature Importance by Wavenumber (Logistic Regression)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Importance (|Weight|)")
    plt.grid(True)

    for x, y in zip(top_wavenumbers, top_importance):
        plt.annotate(f'{x:.0f}', xy=(x, y), xytext=(x, y + 0.01), fontsize=10, ha='center',
                     arrowprops=dict(arrowstyle='->', color='blue', lw=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Feature_Importance.png'))
    plt.close()

# # --- t-SNE Plot ---
# def plot_tsne(X_tsne, y, labels, save_path):
#     plt.figure(figsize=(10, 8))
#     cmap = plt.get_cmap('tab20')
#     num_classes = len(np.unique(y))

#     for group_id in np.unique(y):
#         plt.scatter(X_tsne[y == group_id, 0], 
#                     X_tsne[y == group_id, 1], 
#                     label=labels[group_id],
#                     alpha=0.7, 
#                     s=30, 
#                     color=cmap(group_id % 20))  # tab20 supports 20 distinct colors

#     plt.title("2D t-SNE Projection")
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, 'tSNE_2D.png'), bbox_inches='tight')
#     plt.close()


# --- Main Pipeline ---
def main():
    folders = [
        "afterCol100Pep0",
        "afterCol90Pep10",
        "afterCol80Pep20",
        "afterCol70Pep30",
        "afterCol60Pep40"
    ]

    # Two different root paths
    root_paths = [
        "/Volumes/TIANYI/spec_res/06122025_AUPILLAR_ETCHED_MEM/950-1200",
        "/Volumes/TIANYI/spec_res/07162025_AUPILLAR_ETCHED_MEM/950-1200"
    ]

    save_path = "/Volumes/TIANYI/spec_res/tsne_result_combined"
    os.makedirs(save_path, exist_ok=True)

    data, labels = [], []
    label_counter = 0
    label_names = []

    for root_path in root_paths:
        for folder in folders:
            full_path = os.path.join(root_path, folder, "LMR_1/second_derivative_spec")
            spectra = load_npy_data(full_path, max_samples=2000)
            data.extend(spectra)
            labels.extend([label_counter] * len(spectra))
            # st()
            label_name = os.path.basename(os.path.dirname(root_path)) + "_" + os.path.basename(root_path) + "_" + folder
            label_names.append(label_name)
            print(f"Loaded {len(spectra)} samples from {label_name}")
            label_counter += 1

    # Preprocess
    X = normalize_spectra_zscore(data)
    y = np.array(labels)
    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:4000]
    y_small = y_small[:4000]

    # Dimensionality Reduction
    X_pca = PCA(n_components=50).fit_transform(X_small)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)
    X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_pca)

    # Feature Importance
    clf = LogisticRegression(penalty='l2', solver='liblinear')
    clf.fit(X_small, y_small)
    importance = np.mean(np.abs(clf.coef_), axis=0)
    wavenumbers = np.linspace(950, 1200, len(importance))  # Adjust if needed based on actual range
    plot_feature_importance(importance, wavenumbers, save_path)

    def plot_embedding(embedding, name, coords_label):
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab20')

        # Step 1: Build base group → color index map
        base_groups = sorted(set([lbl.split('_')[-1] for lbl in label_names]))
        base_color_map = {base: 2 * i for i, base in enumerate(base_groups)}

        for i, label in enumerate(np.unique(y_small)):
            full_label = label_names[label]  # e.g., "950-1200_afterCol100Pep0"
            folder_prefix = full_label.split('_')[0]  # e.g., "950-1200"
            base_label = full_label.split('_')[-1]    # e.g., "afterCol100Pep0"

            base_color_index = base_color_map[base_label]

            # Decide color: even index for one folder, odd for the other
            if folder_prefix == '950-1200':  # or whichever is your "first"
                color_index = base_color_index
            else:
                color_index = base_color_index + 1

            plt.scatter(embedding[y_small == label, 0], embedding[y_small == label, 1],
                        label=full_label, alpha=0.5, s=30, color=cmap(color_index))

        plt.title(f"{name} Projection")
        plt.xlabel(f"{coords_label} 1")
        plt.ylabel(f"{coords_label} 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{name}_zscore.png'), bbox_inches='tight')
        plt.show()
        plt.close()


    # Plot t-SNE
    # plot_tsne(X_tsne, y_small, labels=label_names, save_path=save_path)
    plot_embedding(X_tsne, 't-SNE', 't-SNE')
    plot_embedding(X_umap, 'UMAP', 'UMAP')
    print("✅ Analysis completed and plots saved.")

if __name__ == '__main__':
    main()
