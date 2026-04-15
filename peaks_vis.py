import os
import pandas as pd
import matplotlib.pyplot as plt


folder_list = ['liver_ff','liver_ffpe', 'kidney_ff','kidney_ffpe'] #'liver_ff','liver_ffpe', 'kidney_ff','kidney_ffpe'
for f in folder_list:
    # Folder where your CSV files are stored
    folder = f"D:/res/rat_peaks/summary/{f}/"

    # List of HMT CSV files
    hmt_files = ["peak_occurrence_HMT_1.csv", "peak_occurrence_HMT_2.csv", "peak_occurrence_HMT_3.csv", "peak_occurrence_HMT_4.csv", "peak_occurrence_HMT_5.csv", "peak_occurrence_HMT_6.csv"]

    # Target wavenumbers and their window range (+/- delta)
    target_wavenumbers = [968,1036,1082,1168,1238,1310,1400,1448,1464,1516,1546,1658,1742,1788]
    delta = 4  # this will sum from wavenumber-delta to wavenumber+delta
    # Colors for each file
    # Colors for scatter points
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    # Collect occurrences for all files and all wavenumbers
    all_occurrences = {wn: [] for wn in target_wavenumbers}

    for file in hmt_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        
        if 'wavenumber' not in df.columns or 'occurrence' not in df.columns:
            df.columns = ['wavenumber', 'occurrence']
        
        for wn in target_wavenumbers:
            mask = (df['wavenumber'] >= wn - delta) & (df['wavenumber'] <= wn + delta)
            total_occurrence = df.loc[mask, 'occurrence'].sum()
            all_occurrences[wn].append(total_occurrence)

    # Prepare data for plotting
    scatter_x = []
    scatter_y = []
    scatter_colors = []

    for i, wn in enumerate(target_wavenumbers):
        for j, val in enumerate(all_occurrences[wn]):
            # Add slight horizontal jitter so points don't overlap with boxplot
            scatter_x.append(wn + (j - 2.5)*2)  # adjust spacing
            scatter_y.append(val)
            scatter_colors.append(colors[j])

    # Create figure
    plt.figure(figsize=(8,3))

    # Box plot
    plt.boxplot([all_occurrences[wn] for wn in target_wavenumbers],
                positions=target_wavenumbers, widths=14, patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.7),
                medianprops=dict(color='black'))

    # Scatter plot on top
    plt.scatter(scatter_x, scatter_y, color=scatter_colors, s=60, zorder=2)

    # plt.xlabel("Wavenumber (cm⁻¹)",fontsize=14)
    plt.ylabel("Occurrence",fontsize=12)
    plt.xticks(fontsize=12,rotation=30)
    plt.yticks(fontsize=14)
    plt.ylim([-5,110])
    # plt.legend(loc='upper left', fontsize=16)
    # plt.title(f"{f} Occurrence ",fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(folder+f"/{f}_peaks.png"), dpi=300)
    # plt.grid(True)
    
    plt.show()