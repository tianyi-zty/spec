import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Step 1: Define your two folders ---
folder_group1 = "../res/caf2_06132025/tnse/2nd/1000/LMT_3"  # e.g., "./group1"
folder_group2 = "../res/caf2_06132025/tnse/2nd/8020/LMT_5"  # e.g., "./group2"

# --- Step 2: Load and flatten data ---
def load_npy_data(folder):
    data = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.npy'):
            arr = np.load(os.path.join(folder, file))
            if np.any(arr):  # skip all-zero arrays
                data.append(arr.flatten())
    return data

data1 = load_npy_data(folder_group1)
data2 = load_npy_data(folder_group2)

normalized_data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1) + 1e-8)
normalized_data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2) + 1e-8)

# --- Step 3: Combine data and create labels ---
X = np.concatenate((normalized_data1, normalized_data2), axis=0)
y = np.array([0] * len(normalized_data1) + [1] * len(normalized_data2))  # 0 = group1, 1 = group2

# --- Step 4: Run t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# --- Step 5: Plot ---
plt.figure(figsize=(8, 6))
colors = ['royalblue', 'darkorange']
labels = ['1000', '8020']

for group_id in [0, 1]:
    plt.scatter(X_tsne[y == group_id, 0], X_tsne[y == group_id, 1],
                label=labels[group_id], alpha=0.3, s=30, color=colors[group_id])

plt.title("t-SNE of Two Groups")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
