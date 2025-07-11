import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import umap
from pdb import set_trace as st
from collections import defaultdict
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.mplot3d import Axes3D




# ---------------------- #
# Data Loading Functions #
# ---------------------- #

def load_npy_data(folder, max_samples=1000):
    """Load .npy files from a folder and randomly sample up to `max_samples`."""
    print(f"Processing folder: {folder}")
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    if len(files) == 0:
        print(f"⚠️ Nothing in this folder: {folder}")
        return np.array([])  # Return empty array
    
    np.random.seed(42)
    files = np.random.choice(files, size=min(max_samples, len(files)), replace=False)

    data = []
    for file in files:
        arr = np.load(os.path.join(folder, file))
        if np.any(arr):  # skip all-zero arrays
            data.append(arr.flatten())
    return np.array(data)

def normalize_spectra_zscore(X):
    """Apply z-score normalization per spectrum."""
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    return (X - X_mean) / (X_std + 1e-8)

# ---------------------- #
# Main Analysis Pipeline #
# ---------------------- #

def main():
    foldername_list = ['1000', '9010','8020', '7030', '6040'] #'1000B','1000', , '9010', '8020', '7030', '6040'
    filename_list = ['LMT_1','LMT_4'] # 'LMT_2','LMT_3',
    
    base_path = "../res/Caf2_07032025_amide1/"
    save_path = "../res/Caf2_07032025_amide1/result/"
    # representative_save_path = os.path.join(save_path, "representative_spectra")
    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(representative_save_path, exist_ok=True)

    all_data = []
    all_labels = []
    label_names = []
    label_index = 0


    for foldername in foldername_list:
        for filename in filename_list:
            folder_path = os.path.join(base_path, foldername, filename)
            if not os.path.isdir(folder_path):
                continue
            data = load_npy_data(folder_path, max_samples=5000)
            if len(data) == 0:
                continue

            norm_data = normalize_spectra_zscore(data)
            all_data.append(norm_data)
            all_labels += [label_index] * len(norm_data)
            label_names.append(f"{foldername}_{filename}")
            print(f"Loaded {len(norm_data)} data from {foldername}/{filename} as label {label_index}")
            label_index += 1

    # Combine and shuffle
    X = np.concatenate(all_data)
    y = np.array(all_labels)
    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:5000]
    y_small = y_small[:5000]

    # -------------------- #
    # PCA + t-SNE + UMAP  #
    # -------------------- #
    X_pca = PCA(n_components=50).fit_transform(X_small)
    
    X_tsne = TSNE(n_components=2, perplexity=200, random_state=42).fit_transform(X_pca)
    # X_tsne_3d = TSNE(n_components=3, perplexity=200, random_state=42).fit_transform(X_pca)
    X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_pca)
    # X_lda = LDA(n_components=2).fit_transform(X_pca, y_small)
    # --------------------------- #
    # Feature Importance (LogReg) #
    # --------------------------- #
    clf = LogisticRegression(penalty='l2', solver='liblinear')
    clf.fit(X_small, y_small)
    importance = np.mean(np.abs(clf.coef_), axis=0)
    wavenumbers = np.linspace(950, 1800, len(importance))
    top_indices = np.argsort(importance)[-20:]
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
    # t-SNE and UMAP Plots #
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

    plot_embedding(X_tsne, 't-SNE', 't-SNE')
    plot_embedding(X_umap, 'UMAP', 'UMAP')
    # plot_embedding(X_lda, 'LDA', 'LDA')

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # cmap = plt.get_cmap('tab20')

    # for i, label in enumerate(np.unique(y_small)):
    #     ax.scatter(
    #         X_tsne_3d[y_small == label, 0],
    #         X_tsne_3d[y_small == label, 1],
    #         X_tsne_3d[y_small == label, 2],
    #         label=label_names[label],
    #         alpha=0.6,
    #         s=30,
    #         color=cmap(i)
    #     )

    # ax.set_title("3D t-SNE Projection")
    # ax.set_xlabel("t-SNE 1")
    # ax.set_ylabel("t-SNE 2")
    # ax.set_zlabel("t-SNE 3")
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, "tSNE_3D_zscore.png"))
    # plt.show()



    # ---------- Step 1: Build dynamic output folder names ----------
    output_base = "../res/Caf2_07032025_amide1/clustered_tsne"
    output_dirs = {}
    label_names = []
    label_index = 0

    for foldername in foldername_list:
        for filename in filename_list:
            label_name = f"{foldername}_{filename}"
            label_names.append(label_name)
            out_path = os.path.join(output_base, label_name)
            output_dirs[label_index] = out_path
            os.makedirs(out_path, exist_ok=True)
            label_index += 1

    # ---------- Step 2: KMeans clustering on t-SNE and align clusters ----------
    n_clusters = len(np.unique(y_small))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_tsne)
    tsne_labels = kmeans.labels_

    # Hungarian alignment
    conf_mat = confusion_matrix(y_small, tsne_labels)
    print("🔍 Confusion matrix (rows=true, cols=clusters):\n", conf_mat)
    # ---------- Step 2.5: Plot and save confusion matrix ----------
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Cluster {i}" for i in range(n_clusters)],
                yticklabels=[label_names[i] for i in range(n_clusters)])
    plt.xlabel("KMeans Cluster Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (t-SNE KMeans vs. True Labels)")
    plt.tight_layout()

    confmat_path = os.path.join(save_path, "Confusion_Matrix_tsne.png")
    plt.savefig(confmat_path)
    plt.close()
    print(f"📊 Confusion matrix saved to {confmat_path}")
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    label_map = {col: row for row, col in zip(col_ind, row_ind)}
    mapped_tsne_labels = np.array([label_map[lbl] for lbl in tsne_labels])

    # # ---------- Step 3: Save correctly clustered spectra ----------
    # count = defaultdict(int)
    # N = len(y_small)

    # for i in range(N):
    #     true_label = y_small[i]
    #     predicted_label = mapped_tsne_labels[i]
    #     if true_label == predicted_label and true_label in output_dirs:
    #         folder = output_dirs[true_label]
    #         filename = f"spectrum_{true_label}_{count[true_label]:05d}.npy"
    #         np.save(os.path.join(folder, filename), X_small[i])
    #         count[true_label] += 1

    # # ---------- Step 4: Print summary ----------
    # print("✅ Saved spectra per correctly clustered group:")
    # for label, c in count.items():
    #     print(f"  {label_names[label]}: {c} spectra")


if __name__ == '__main__':
    main()
