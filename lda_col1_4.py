import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

# ---------------------- #
# Main Analysis Pipeline #
# ---------------------- #

def main():
    foldername_list = ['1000', '9109', '9505']
    filename_list = ['LMT_1', 'LMT_2', 'LMT_3']

    base_path = "../res/04082026_col1+4/CAF2/org"
    save_path = "../res/04082026_col1+4/CAF2/org/result/"
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

            data = load_npy_data(folder_path, max_samples=512)
            if len(data) == 0:
                continue

            all_data.append(data)
            all_labels += [label_index] * len(data)
            label_names.append(f"{foldername}_{filename}")
            print(f"Loaded {len(data)} data from {foldername}_{filename} as label {label_index}")
            label_index += 1

    if len(all_data) == 0:
        print("No data loaded.")
        return

    X = np.concatenate(all_data, axis=0)
    y = np.array(all_labels)

    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:1000]
    y_small = y_small[:1000]

    # -------------------- #
    # LDA projection       #
    # -------------------- #
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_small, y_small)

    # --------------------------- #
    # Feature Importance (LDA)    #
    # --------------------------- #
    lda_clf = LDA()
    lda_clf.fit(X_small, y_small)

    if hasattr(lda_clf, "coef_"):
        importance = np.mean(np.abs(lda_clf.coef_), axis=0)
        wavenumbers = np.linspace(950, 1800, len(importance))
        top_indices = np.argsort(importance)[-20:]
        top_wavenumbers = wavenumbers[top_indices]
        top_importance = importance[top_indices]

        plt.figure(figsize=(12, 5))
        plt.plot(wavenumbers, importance, linewidth=1.5)
        plt.title("Feature Importance by Wavenumber (LDA)")
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Importance (|Coefficient|)")
        plt.grid(True)

        for x, y_val in zip(top_wavenumbers, top_importance):
            plt.annotate(
                f"{x:.0f}",
                xy=(x, y_val),
                xytext=(x, y_val + np.max(importance) * 0.02),
                fontsize=8,
                ha='center'
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "Feature_Importance_LDA.png"), dpi=300)
        plt.show()
        plt.close()

    def plot_embedding(embedding, name, coords_label, foldername_list, filename_list):
        fig, ax = plt.subplots(figsize=(10, 8))

        folder_colors = {
            '1000': 'tab:blue',
            '9109': 'tab:orange',
            '9505': 'tab:green',
        }

        filename_alphas = {
            'LMT_1': 1.0,
            'LMT_2': 0.7,
            'LMT_3': 0.4,
        }

        for folder_idx, foldername in enumerate(foldername_list):
            for file_idx, filename in enumerate(filename_list):
                label = folder_idx * len(filename_list) + file_idx
                alpha = filename_alphas.get(filename, 0.5)
                color = folder_colors.get(foldername, 'gray')
                indices = np.where(y_small == label)[0]

                if len(indices) == 0:
                    continue

                ax.scatter(
                    embedding[indices, 0],
                    embedding[indices, 1],
                    label=f'{foldername}_{filename}',
                    alpha=alpha,
                    s=30,
                    color=color
                )

        ax.set_title(f"{name} Projection")
        ax.set_xlabel(f"{coords_label} 1")
        ax.set_ylabel(f"{coords_label} 2")
        ax.grid(True)

        # Legend outside on the right
        ax.legend(
            fontsize=8,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )

        plt.subplots_adjust(right=0.78)
        plt.savefig(os.path.join(save_path, f'{name}_zscore.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    plot_embedding(X_lda, 'LDA', 'LD', foldername_list, filename_list)

if __name__ == '__main__':
    main()