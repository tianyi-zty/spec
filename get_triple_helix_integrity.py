import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os



# --- Load your CSV file ---
folder_path = '../res/Caf2_09022025_amide1/second_derivative/'
file_name = ['1000', '9010', '8020', '7030', '6040']
for file in file_name:
    print(file)
    path = folder_path + file
    df = pd.read_csv(path + '/integral_value_summary.csv')  # Change to your file name if different

    # Mapping of index to band names (for clarity, optional)
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

    # Extract mean values using index
    phosphate_I_AmideIII = df.loc[3, 'mean']
    asym_CH3_bending = df.loc[7, 'mean']

    beta = df.loc[10, 'mean']
    alpha = df.loc[11, 'mean']
    coil = df.loc[12, 'mean']
    total_amide_I = beta + alpha + coil

    # Calculate the desired ratios
    ratio_1 = phosphate_I_AmideIII / asym_CH3_bending
    ratio_beta = beta / total_amide_I
    ratio_alpha = alpha / total_amide_I
    ratio_coil = coil / total_amide_I

    # Print the results
    print("Ratio (phosphate I / Amide III) / Asym CH3 bending:", f'{ratio_1:.2f}')
    print("Beta-sheet Amide I / Total Amide I:", f'{ratio_beta:.2f}')
    print("Alpha-helix Amide I / Total Amide I:", f'{ratio_alpha:.2f}')
    print("Coils/turn Amide I / Total Amide I:", f'{ratio_coil:.2f}')
