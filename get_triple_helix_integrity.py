import pandas as pd
import numpy as np
from pdb import set_trace as st
import re

# --- Load the combined CSV ---
csv_path = '../res/Caf2_10302025/spectrum_fitting_results/subspectrum_fitting_results.csv'
# Read the file properly by splitting on commas
df = pd.read_csv(csv_path, sep=",", engine="python")
# --- Mapping of peak index to band names ---
band_names = {
    0: "Symmetric PO2 stretching",
    1: "Phosphate band/Collagen",
    2: "Amide III",
    3: "phosphate I/Amide III",
    4: "Amide III band components of protein",
    5: "CH2 wagging/collagen",
    6: "Symmetric CH3 bending",
    7: "Asymmetric CH3 bending",
    8: "Amide II beta-sheet",
    9: "Amide II",
    10: "beta-sheet Amide I",
    11: "alpha-helix Amide I",
    12: "Coils/turn Amide I"
}
# st()
# --- Group by File Name ---
grouped = df.groupby('File Name')

# --- Prepare results ---
results = []

for name, group in grouped:
    group = group.reset_index(drop=True)
    
    try:
        phosphate_I_AmideIII = group.loc[3, 'Integral Value']
        asym_CH3_bending = group.loc[7, 'Integral Value']

        beta = group.loc[10, 'Integral Value']
        alpha = group.loc[11, 'Integral Value']
        coil = group.loc[12, 'Integral Value']
        total_amide_I = beta + alpha + coil

        ratio_1 = phosphate_I_AmideIII / asym_CH3_bending
        ratio_beta = beta / total_amide_I
        ratio_alpha = alpha / total_amide_I
        ratio_coil = coil / total_amide_I

        results.append({
            'File Name': name,
            'Triple helix integrity': ratio_1,
            'Beta-sheet': ratio_beta,
            'Alpha-helix': ratio_alpha,
            'Beta-turn': ratio_coil
        })
    
    except Exception as e:
        print(f"⚠️ Skipped {name}: {e}")

# --- Convert to DataFrame and save ---
results_df = pd.DataFrame(results)
results_df.to_csv('../res/Caf2_10302025/ratio_summary.csv', index=False)

# print(results_df)

# Extract the sample group key (e.g., "1000_LMT", "6040PSER_LMT", etc.)
results_df['Group'] = results_df['File Name'].apply(
    lambda x: re.sub(r'_mean_spectrum\.npy$', '', x)  # remove suffix
)
results_df['Group'] = results_df['Group'].apply(
    lambda x: '_'.join(x.split('_')[:2])  # keep first two parts, e.g. "1000_LMT" or "9010PSER_LMT"
)

# --- Group by the new key and average ---
avg_df = results_df.groupby('Group', as_index=False)[
    ['Triple helix integrity', 'Beta-sheet', 'Alpha-helix', 'Beta-turn']
].mean()
# --- Round to 2 decimal places ---
avg_df = avg_df.round(2)

# --- Save and print ---
avg_df.to_csv('../res/Caf2_10302025/ratio_summary_grouped.csv', index=False)
print(avg_df)

    
