import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Set the folder and file path ----
csv_folder = "../res/Caf2_10032025/bulk_comp/second_derivative/pser/"  # 🔹 change this to your folder path
csv_file = os.path.join(csv_folder, "peak_occurrence_all.csv")         # name of the combined file

# ---- Load the CSV ----
df = pd.read_csv(csv_file)

# Ensure correct column names
df.columns = ["Wavenumber", "Occurrence"]

# ---- Define x-axis range (950–1800, step=2) ----
x_axis = np.arange(950, 1802, 2)
y_axis = np.zeros_like(x_axis, dtype=float)

# Fill in occurrences where wavenumber matches
for _, row in df.iterrows():
    if row["Wavenumber"] in x_axis:
        idx = np.where(x_axis == row["Wavenumber"])[0][0]
        y_axis[idx] = row["Occurrence"]

# ---- Plot ----
plt.figure(figsize=(12, 6))
plt.bar(x_axis, y_axis, width=2, color='steelblue', edgecolor='k')

plt.xlabel("Wavenumber (cm⁻¹)", fontsize=12)
plt.ylabel("Occurrence", fontsize=12)
plt.title("Peak Occurrence vs Wavenumber", fontsize=14)
plt.xlim(950, 1800)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(csv_folder+f"/peaks.png"), dpi=300)
plt.show()
