import pandas as pd
import numpy as np
import glob
import os

def load_and_count_all_peaks(csv_files):
    peak_counter = {}

    for file in csv_files:
        print(f"Processing: {file}")
        df = pd.read_csv(file)

        # Drop first column (filename)
        df_numeric = df.iloc[:, 1:]

        # Flatten and drop NaN
        peaks = df_numeric.values.flatten()
        peaks = peaks[~np.isnan(peaks)]

        # Count occurrences
        for peak in peaks:
            peak = float(peak)
            peak_counter[peak] = peak_counter.get(peak, 0) + 1

    # Convert to DataFrame
    peak_counts_df = pd.DataFrame(list(peak_counter.items()), columns=['Wavenumber', 'Occurrence'])
    peak_counts_df = peak_counts_df.sort_values(by='Wavenumber').reset_index(drop=True)

    return peak_counts_df


if __name__ == "__main__":
    # Folder containing all your CSV files
    csv_folder = "../res/Caf2_10032025/bulk_comp/second_derivative/col/"
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    print(f"Found {len(csv_files)} CSV files.")

    # Combine all CSVs together for global counting
    peak_counts_df = load_and_count_all_peaks(csv_files)
    print(peak_counts_df)

    # Save combined results
    save_path = csv_folder
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, "peak_occurrence_all.csv")
    peak_counts_df.to_csv(out_file, index=False)

    print(f"✅ Saved total peak occurrence to {out_file}")
