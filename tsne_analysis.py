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

# --- Normalization Functions ---
def normalize_spectra_zscore(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)

# --- Data Loader ---
def load_npy_data(folder, max_samples=1000):
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

# --- t-SNE Plot ---
def plot_tsne(X_tsne, y, labels, save_path):
    plt.figure(figsize=(10, 8))
    colors = ['royalblue', 'darkorange', 'crimson', 'seagreen', 'mediumvioletred']
    for group_id in np.unique(y):
        plt.scatter(X_tsne[y == group_id, 0], X_tsne[y == group_id, 1], label=labels[group_id],
                    alpha=0.6, s=30, color=colors[group_id])
    plt.title("2D t-SNE Projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'tSNE_2D.png'))
    plt.close()

# --- Main Pipeline ---
def main():
    folders = [
        "afterCol100Pep0",
        "afterCol90Pep10",
        "afterCol80Pep20",
        "afterCol70Pep30",
        "afterCol60Pep40"
    ]
    root_path = "/Volumes/TIANYI/spec_res/07162025_AUPILLAR_ETCHED_MEM/950-1200"
    save_path = os.path.join(root_path, "tsne_result") #_second_derivative
    os.makedirs(save_path, exist_ok=True)

    data, labels = [], []
    for i, folder in enumerate(folders):
        full_path = os.path.join(root_path, folder, "LMR1/filtered_spec") #second_derivative_spec filtered_spec
        spectra = load_npy_data(full_path, max_samples=1000)
        data.extend(spectra)
        labels.extend([i] * len(spectra))
        print(f"Loaded {len(spectra)} samples from {folder}")

    # Preprocess
    X = normalize_spectra_zscore(data)
    y = np.array(labels)
    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:2000]
    y_small = y_small[:2000]

    # Dimensionality Reduction
    X_pca = PCA(n_components=50).fit_transform(X_small)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)

    # Feature Importance
    clf = LogisticRegression(penalty='l2', solver='liblinear')
    clf.fit(X_small, y_small)
    importance = np.mean(np.abs(clf.coef_), axis=0)
    wavenumbers = np.linspace(950, 1200, len(importance))
    plot_feature_importance(importance, wavenumbers, save_path)

    # Plot t-SNE
    plot_tsne(X_tsne, y_small, labels=['1000', '9010', '8020', '7030', '6040'], save_path=save_path)
    print("✅ Analysis completed and plots saved.")

if __name__ == '__main__':
    main()
