import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.utils import shuffle
from pdb import set_trace as st
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# from MulticoreTSNE import MulticoreTSNE as TSNE

#Min-Max Normalization (0 to 1 per spectrum)
def normalize_spectra_minmax(X):
    X_min = np.array(X).min(axis=1, keepdims=True)
    X_max = np.array(X).max(axis=1, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)
#Z-score Normalization (mean = 0, std = 1 per spectrum)
def normalize_spectra_zscore(X):
    X_mean = np.array(X).mean(axis=1, keepdims=True)
    X_std = np.array(X).std(axis=1, keepdims=True)
    return (X - X_mean) / (X_std + 1e-8)

# --- Step 2: Load and flatten data ---
def load_npy_data(folder):
    data = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.npy'):
            arr = np.load(os.path.join(folder, file))
            if np.any(arr):  # skip all-zero arrays
                data.append(arr.flatten())
                # st()
    return data

def load_npy_data(folder, max_samples=1000):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    np.random.seed(42)
    files = np.random.choice(files, size=min(max_samples, len(files)), replace=False)

    data = []
    for file in files:
        arr = np.load(os.path.join(folder, file))
        if np.any(arr):  # skip all-zero arrays
            data.append(arr.flatten())
    return data
 
def main():

    # --- Step 1: Define your two folders ---
    folder_group1 = "../res/Caf2_06232025_tnse/original/1000/LMT_3"  
    folder_group2 = "../res/Caf2_06232025_tnse/original/1000/LMT_4" 
    folder_group3 = "../res/Caf2_06232025_tnse/original/9010/LMT_2" 
    folder_group4 = "../res/Caf2_06232025_tnse/original/9010/LMT_3" 
    folder_group5 = "../res/Caf2_06232025_tnse/original/8020/LMT_1" 
    folder_group6 = "../res/Caf2_06232025_tnse/original/8020/LMT_2" 
    folder_group7 = "../res/Caf2_06232025_tnse/original/7030/LMT_1" 
    folder_group8 = "../res/Caf2_06232025_tnse/original/7030/LMT_2" 
    folder_group9 = "../res/Caf2_06232025_tnse/original/6040/LMT_1" 
    folder_group10 = "../res/Caf2_06232025_tnse/original/6040/LMT_2" 
    

    save_path =  "../res/Caf2_06232025_tnse/result_ori/" 
    os.makedirs(save_path, exist_ok=True)

    data1 = load_npy_data(folder_group1, max_samples=1000)
    data2 = load_npy_data(folder_group2, max_samples=1000)
    data3 = load_npy_data(folder_group3, max_samples=1000)
    data4 = load_npy_data(folder_group4, max_samples=1000)
    data5 = load_npy_data(folder_group5, max_samples=1000)
    data6 = load_npy_data(folder_group6, max_samples=1000)
    data7 = load_npy_data(folder_group7, max_samples=1000)
    data8 = load_npy_data(folder_group8, max_samples=1000)
    data9 = load_npy_data(folder_group9, max_samples=1000)
    data10 = load_npy_data(folder_group10, max_samples=1000)


    for i, d in enumerate([data1, data2, data3, data4, data5, data6]):
        print(f"Group {i}: {len(d)} samples")

    normalized_data1 = normalize_spectra_zscore(data1)
    normalized_data2 = normalize_spectra_zscore(data2)
    normalized_data3 = normalize_spectra_zscore(data3)
    normalized_data4 = normalize_spectra_zscore(data4)
    normalized_data5 = normalize_spectra_zscore(data5)
    normalized_data6 = normalize_spectra_zscore(data6)
    normalized_data7 = normalize_spectra_zscore(data7)
    normalized_data8 = normalize_spectra_zscore(data8)
    normalized_data9 = normalize_spectra_zscore(data9)
    normalized_data10 = normalize_spectra_zscore(data10)

    # --- Step 3: Combine data and create labels ---
    X = np.concatenate((normalized_data1, normalized_data2, normalized_data3, normalized_data4, normalized_data5, 
                        normalized_data6, normalized_data7, normalized_data8, normalized_data9, normalized_data10), axis=0)
    y = np.array([0]*len(normalized_data1) + [1]*len(normalized_data2) + [2]*len(normalized_data3)+[3]*len(normalized_data4) +
                  [4]*len(normalized_data5) + [5]*len(normalized_data6)+[6]*len(normalized_data7) + [7]*len(normalized_data8) + 
                  [8]*len(normalized_data9)+ [9]*len(normalized_data10))  # 0 = group1, 1 = group2

    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:2000]
    y_small = y_small[:2000]
    # st()

    # Reduce to 50 components or fewer before t-SNE
    X_pca = PCA(n_components=50).fit_transform(X_small)
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    # X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_pca)

    #######feature importance using logistic regression
    # Step 4a: Train a logistic regression model to identify separating features
    clf = LogisticRegression(penalty='l2', solver='liblinear')
    clf.fit(X_small, y_small)

    # Step 4b: Get feature importance (absolute weights)
    importance = np.mean(np.abs(clf.coef_), axis=0)
    # Map indices to wavenumbers
    wavenumbers = np.linspace(950, 1800, len(importance))

    # Sort and get top 10 important features
    top_indices = np.argsort(importance)[-20:]
    top_wavenumbers = wavenumbers[top_indices]
    top_importance = importance[top_indices]

    # Plot
    plt.figure(figsize=(24, 15))
    plt.plot(wavenumbers, importance, color='darkred', linewidth=1.5)
    plt.title("Feature Importance by Wavenumber (Logistic Regression)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Importance (|Weight|)")
    plt.grid(True)

    # Annotate top 10 peaks
    for x, y in zip(top_wavenumbers, top_importance):
        plt.annotate(f'{x:.0f}', xy=(x, y), xytext=(x, y + 0.01),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'Feature Importance.png'))
    plt.show()
    plt.close()

    # # # --- Step 5: Plot ---
    plt.figure(figsize=(10, 8))
    colors = ['royalblue', 'darkorange', 'crimson', 'seagreen', 'mediumvioletred', 'olive','dodgerblue','indigo','firebrick','peru' ]
    labels = ['1000LMT_3', '1000LMT_4', '9010LMT_2', '9010LMT_3','8020LMT_1', '8020LMT_2', '7030LMT_1', '7030LMT_2','6040LMT_1', '6040LMT_2']

    for group_id in [0, 1, 2, 3, 4, 5,6,7,8,9]:  # or [0, 1, 2] if you want all 3
        plt.scatter(X_tsne[y_small == group_id, 0],
                    X_tsne[y_small == group_id, 1],
                    label=labels[group_id],
                    alpha=0.5, s=30, color=colors[group_id])

    plt.title("2D t-SNE Projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f't-SNE_2D_zscore.png'))
    plt.show()
    plt.close()
    # st()
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # colors = ['royalblue', 'darkorange', 'crimson']
    # labels = ['1000', '8020', '6040']

    # for group_id in [0, 2]:
    #     ax.scatter(X_tsne[y_small == group_id, 0],
    #             X_tsne[y_small == group_id, 1],
    #             X_tsne[y_small == group_id, 2],
    #             label=labels[group_id],
    #             alpha=0.3, s=30, color=colors[group_id])

    # ax.set_title("t-SNE of Two Groups")
    # ax.set_xlabel("t-SNE 1")
    # ax.set_ylabel("t-SNE 2")
    # ax.set_zlabel("t-SNE 3")
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, f't-SNE_zscore.png'))
    # plt.show()
    # plt.close()

    # # --- Step 5: Plot ---
    # plt.figure(figsize=(8, 6))
    # colors = ['mediumseagreen', 'crimson']
    # labels = ['1000', '8020']

    # for group_id in [0, 1]:
    #     plt.scatter(X_pca[y_small == group_id, 0], X_umap[y_small == group_id, 1],
    #                 label=labels[group_id], alpha=0.3, s=30, color=colors[group_id])

    # plt.title("PCA of Two Groups")
    # plt.xlabel("PCA 1")
    # plt.ylabel("PCA 2")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, f'PCA_zscore.png'))
    # plt.show()

    # # --- Step 5: Plot ---
    # plt.figure(figsize=(8, 6))
    # colors = ['mediumpurple', 'gold']
    # labels = ['1000', '8020']

    # for group_id in [0, 1]:
    #     plt.scatter(X_umap[y_small == group_id, 0], X_tsne[y_small == group_id, 1],
    #                 label=labels[group_id], alpha=0.3, s=30, color=colors[group_id])

    # plt.title("Umap of Two Groups")
    # plt.xlabel("Umap 1")
    # plt.ylabel("Umap 2")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, f'Umap_zscore.png'))
    # plt.show()


    # ############# save the data results from tsne to two folders
    # output_dirs = {0: "../res/Caf2_06232025_tnse/result/clustered_original/1000LMT_1", 1: "../res/Caf2_06232025_tnse/result/clustered_original/1000LMT_3"
    #             , 2: "../res/Caf2_06232025_tnse/result/clustered_original/1000-polyfiber"}
    # for folder in output_dirs.values():
    #     os.makedirs(folder, exist_ok=True)

    # # Step 2: Infer t-SNE-based labels (e.g., from x-axis threshold)
    # # KMeans clustering
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(X_tsne)
    # tsne_labels = kmeans.labels_

    # # Print confusion matrix before alignment
    # print("Confusion matrix (rows: true labels, cols: cluster labels):")
    # conf_mat = confusion_matrix(y_small, tsne_labels)
    # print(conf_mat)

    # # Align clusters to true labels using Hungarian algorithm
    # row_ind, col_ind = linear_sum_assignment(-conf_mat)
    # label_map = {col: row for row, col in zip(col_ind, row_ind)}

    # # Remap t-SNE cluster labels to best-matching true labels
    # mapped_tsne_labels = np.array([label_map[label] for label in tsne_labels])
    # st()
    # # Step 3: Save spectra where true label matches t-SNE label
    # count = {0: 0, 1: 0, 2: 0}
    # N = len(y_small)
    # for i in range(N):
    #     if y_small[i] == mapped_tsne_labels[i]:
    #         group = y_small[i]
    #         folder = output_dirs[group]
    #         filename = f"spectrum_{group}_{count[group]:05d}.npy"
    #         save_path = os.path.join(folder, filename)
    #         np.save(save_path, X[i])
    #         count[group] += 1


    # print(f"✅ Saved {count[0]} spectra to '1000/', and {count[1]} spectra to '8020/', and {count[2]} spectra to '6040/'")

if __name__ == '__main__':
    main()
