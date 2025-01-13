import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace as st
import os

# Load your CSV data
save_path = '../res/AuPillars_Al2O3_12102024/1/pixel/95-5/'
file_name = 'subspectrum_fitting_results_1'
data = pd.read_csv(save_path+file_name+'.csv')  # Update with the correct file path
# st()

# Check the number of unique 'File Name' entries
num_unique_files = len(data['File Name'].unique())
print(f"Number of unique files: {num_unique_files}")

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Define a colormap with 8 colors for the 8 rows
colors = plt.cm.viridis(np.linspace(0, 1, 8))
labels = ['CH3 of collagen','Amide II','Amide II b-sheet', 'Amide II', 'DNA', 'Amide I b-sheet', 'Amide I a-helix', 'Amide I']
# Create a new figure
plt.figure(figsize=(10, 6))

# Iterate through rows by their index within each file
for row_index in range(8):
    # Filter data for the current row index
    row_data = data.groupby('File Name').nth(row_index).reset_index()  # Extract the nth row from each group

    # Scatter plot (Center Value vs Integral Value) using the same color for the current row index
    plt.scatter(row_data['Center Value'], row_data['Integral Value'], color=colors[row_index], 
                label=labels[row_index])

# Set axis labels and title
plt.xlabel('Wavelength (Center Value)', fontsize=14)
plt.ylabel('Integral Value', fontsize=14)
plt.title('Center Value vs Integral Value by Row Index', fontsize=16)

# Show the legend
plt.legend(title='Row Index', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show grid for better readability
plt.grid(True)

# Show the plot
plt.tight_layout()
# plt.show()

plot_file = os.path.join(save_path, f"{file_name}_fitting.png")
plt.savefig(plot_file)
