#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waterfall_3d_fixedcolor_tab20.py
3D 光谱瀑布图
- x 轴数据不变但视觉拉长
- 组颜色固定（tab20）：1000蓝，9010橘，8020绿，7030红，6040紫
- 组内不同 label 用透明度区分
- 平均线到 -1 阴影填充
- 无网格
"""

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
import matplotlib.colors as mcolors
import os

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

# ---------------------- #
# 3D Waterfall Plotting  #
# ---------------------- #
def plot_3d_waterfall(X, y, labels, save_path,
                      x_aspect=3.5, y_aspect=5,
                      line_width=2.0, min_alpha=0.3):
    wavelengths = np.linspace(950, 1800, X.shape[1])
    unique_labels = np.unique(y)

    # tab20 调色板
    tab20 = plt.get_cmap('tab20').colors
    group_colors_order = ['1000', '9010', '8020', '7030', '6040']
    group_colors = {gk: tab20[i*2] for i, gk in enumerate(group_colors_order)}
    # i*4 取 tab20 的 0,4,8,12,16 五个主要颜色

    # 分组函数
    def group_key_from_labelname(name: str) -> str:
        return name.split("_")[0] if "_" in name else name

    # 收集组和组内 label
    group_to_label_indices = defaultdict(list)
    for lab in unique_labels:
        gk = group_key_from_labelname(labels[lab])
        group_to_label_indices[gk].append(lab)

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, lab in enumerate(unique_labels):
        group_data = X[y == lab]
        if group_data.size == 0:
            continue
        avg = np.mean(group_data, axis=0)

        gk = group_key_from_labelname(labels[lab])
        base_color = group_colors.get(gk, tab20[0])  # 默认蓝色

        # 组内第 i 条线，透明度从深到浅
        pos_in_group = group_to_label_indices[gk].index(lab)
        n_lines = len(group_to_label_indices[gk])
        alphas = np.linspace(1.0, min_alpha, n_lines)
        alpha = alphas[pos_in_group]

        color = mcolors.to_rgba(base_color, alpha=alpha)

        # 平均线
        ax.plot(wavelengths, [idx]*len(wavelengths), avg, color=color, linewidth=line_width)

        # 阴影填充到 -1
        zs_lower = np.full_like(avg, -1.0)
        verts = list(zip(wavelengths, [idx]*len(wavelengths), avg)) + \
                list(zip(wavelengths[::-1], [idx]*len(wavelengths), zs_lower[::-1]))
        poly = Poly3DCollection([verts], color=color, alpha=alpha*0.3)
        ax.add_collection3d(poly)

    # 轴与视角
    # ax.set_title(title, fontsize=16)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)",fontsize=10)
    # ax.set_ylabel("Group")
    ax.set_zlabel("Normalized Intensity", fontsize=10)
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels([labels[l] for l in unique_labels], fontsize=10)
    ax.view_init(elev=30, azim=-60)
    ax.grid(False)
    # x 轴刻度字体变小
    ax.tick_params(axis='x', labelsize=8)  # 改 8 为你想要的字号
    ax.tick_params(axis='y', labelsize=8) # y轴刻度字体
    ax.tick_params(axis='z', labelsize=8) # z轴刻度字体

    try:
        ax.set_box_aspect((x_aspect, y_aspect, 1.0))
    except Exception:
        pass

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

# ---------------------- #
# Main
# ---------------------- #
if __name__ == "__main__":

    foldername_list = ['1000', '9010', '8020', '7030', '6040']
    filename_list = ['LMT_2', 'LMT_3']
    folder_map = {'LMT_2': '1', 'LMT_3': '2'}

    base_path = "../../spec_res/Caf2_07182025_amide1/"
    out_path = "../../spec_res/Caf2_07182025_amide1/result-1/3d_waterfall.png"

    all_data, all_labels, label_names = [], [], []
    label_index = 0

    for foldername in foldername_list:
        for filename in filename_list:
            folder_path = os.path.join(base_path, foldername, filename)
            if not os.path.isdir(folder_path):
                continue
            data = load_npy_data(folder_path, max_samples=2000)
            if len(data) == 0:
                continue
            norm_data = normalize_spectra_zscore(data)
            all_data.append(norm_data)
            all_labels += [label_index] * len(norm_data)
            label_names.append(f"{foldername}_{folder_map.get(filename, filename)}")
            print(f"Loaded {len(norm_data)} spectra from {foldername}/{filename} as label {label_index}")
            label_index += 1

    X = np.concatenate(all_data) if len(all_data) else np.empty((0,0))
    y = np.array(all_labels)
    label_names = np.array(label_names)

    if X.size == 0:
        raise RuntimeError("No data loaded.base_path / folder lists.")

    X_small, y_small = shuffle(X, y, random_state=42)
    X_small = X_small[:5000]
    y_small = y_small[:5000]

    plot_3d_waterfall(
        X_small,
        y_small,
        label_names,
        save_path=out_path,
        # title="3D Waterfall Spectrum (Fixed tab20 colors, alpha varies)",
        x_aspect=4,
        y_aspect=6,
        line_width=2.0,
        min_alpha=0.3
    )
