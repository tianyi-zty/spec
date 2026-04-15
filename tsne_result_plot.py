import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

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
def plot_avg_std_with_top_features(
    X,
    y,
    label_map,
    class_to_folder,
    top_feature_indices,
    save_path,
    title
):
    wavelengths = np.linspace(950, 1800, X.shape[1])
    unique_labels = np.unique(y)

    # color by folder/group
    folder_colors = {
        # 'liver_ffpe': 'orange',
        # 'kidney_ffpe': 'blue',
        # 'liver_ff': 'green',
        # 'kidney_ff': 'purple',
        '1000': 'tab:blue',
        '9109': 'tab:orange',
        '9505': 'tab:green',
    }

    # transparency by filename/LMT/HMT
    filename_alpha_map = {
        'LMT_1': 1.0,
        'LMT_2': 0.7,
        'LMT_3': 0.4,
        # 'HMT_1': 1.0,
        # 'HMT_2': 0.7,
        # 'HMT_3': 0.4,
    }

    plt.figure(figsize=(14, 8))

    for label in unique_labels:
        group_data = X[y == label]
        if group_data.size == 0:
            continue

        avg = np.mean(group_data, axis=0)
        std = np.std(group_data, axis=0)

        class_name = label_map[label]              # e.g. kidney_ffpe_HMT_1
        folder_name = class_to_folder[label]       # e.g. kidney_ffpe
        filename = class_name.replace(folder_name + "_", "", 1)

        color = folder_colors.get(folder_name, 'gray')
        alpha_fill = filename_alpha_map.get(filename, 0.2)

        plt.plot(
            wavelengths,
            avg,
            label=class_name,
            color=color,
            linewidth=2.5
        )
        plt.fill_between(
            wavelengths,
            avg - std,
            avg + std,
            color=color,
            alpha=alpha_fill * 0.25
        )

    # mark top features
    y_min, y_max = plt.ylim()
    tick_height = 0.05 * (y_max - y_min)
    for idx in top_feature_indices:
        wn = wavelengths[idx]
        plt.vlines(wn, y_min, y_min + tick_height, color='black', linewidth=1.2)

    plt.title(title, fontsize=16)
    plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    plt.ylabel("Normalized Intensity", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_')}_full_spectrum.png"), dpi=300)
    plt.show()
    plt.close()


# ---------------------- #
# Main Analysis Pipeline #
# ---------------------- #
def main():
    # Example 1: two folders, one filename each
    # foldername_list = ['kidney_ffpe', 'kidney_ff']
    # filename_list = ['HMT_1']

    # Example 2: same group, multiple LMT/HMT treated as different classes
    foldername_list = ['1000','8020']
    filename_list = ['LMT_1','LMT_2','LMT_3']  # can be ['LMT_1', 'LMT_2', 'LMT_3'] or multiple HMT/LMT

    base_path = "../res/03232026_col1+4/CAF2/org"
    save_path = "../res/03232026_col1+4/CAF2/org/result-/"
    os.makedirs(save_path, exist_ok=True)

    all_data = []
    all_labels = []
    label_map = {}         # numeric label -> class name (folder + filename)
    class_to_folder = {}   # numeric label -> folder/group name
    label_index = 0

    for foldername in foldername_list:
        for filename in filename_list:
            folder_path = os.path.join(base_path, foldername, filename)
            if not os.path.isdir(folder_path):
                print(f"⚠️ Folder not found: {folder_path}")
                continue

            data = load_npy_data(folder_path, max_samples=512)
            if len(data) == 0:
                continue

            norm_data = normalize_spectra_zscore(data)
            all_data.append(norm_data)

            class_name = f"{foldername}_{filename}"   # each folder+filename is a different class
            all_labels += [label_index] * len(norm_data)
            label_map[label_index] = class_name
            class_to_folder[label_index] = foldername

            print(f"Loaded {len(norm_data)} data from {class_name} as label {label_index}")
            label_index += 1

    if len(all_data) == 0:
        print("No data loaded. Exiting.")
        return

    X = np.concatenate(all_data, axis=0)
    y = np.array(all_labels)

    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:1000]
    y_small = y_small[:1000]

    print("\nClass mapping:")
    for k in sorted(label_map.keys()):
        print(f"  {k}: {label_map[k]}")

    # --------------------------- #
    # Feature Importance (LogReg) #
    # --------------------------- #
    # multiclass-safe solver
    clf = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        random_state=42
    )
    clf.fit(X_small, y_small)

    # average absolute coefficient across classes
    importance = np.mean(np.abs(clf.coef_), axis=0)
    wavenumbers = np.linspace(950, 1800, len(importance))

    top_n = min(60, len(importance))
    top_indices = np.argsort(importance)[-top_n:]
    top_indices = np.sort(top_indices)
    top_importance = importance[top_indices]

    print("\nTop features:")
    for idx, imp in zip(top_indices, top_importance):
        print(f"Wavenumber {wavenumbers[idx]:.1f} cm^-1 : importance = {imp:.6f}")

    # ------------------------ #
    # Avg ± STD Spectrum Plot #
    # ------------------------ #
    plot_avg_std_with_top_features(
        X_small,
        y_small,
        label_map,
        class_to_folder,
        top_indices,
        save_path,
        title="Avg STD Spectrum with Top Features"
    )


if __name__ == '__main__':
    main()