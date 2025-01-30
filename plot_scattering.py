import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load your CSV data
wv1=1200
filename = f'{wv1}-{wv1+200}'
save_path = r'../res/AuPillars_10nmAl2O3_01162025/2ndafter/'+filename+'/result/'
file_name = 'subspectrum_fitting_results'
data = pd.read_csv(save_path + file_name + '.csv')  # Update with the correct file path

# Check the number of unique 'File Name' entries
num_unique_files = len(data['File Name'].unique())
print(f"Number of unique files: {num_unique_files}")

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Define a colormap with 8 colors for the 8 rows
colornm = 1
colors = plt.cm.viridis(np.linspace(0, 1, colornm))
labels = ["Amide III"]
# "Amide III", "Collagen Amide III","Amide III"

# Scatter Plot
plt.figure(figsize=(10, 6))
for row_index in range(colornm):
    # Filter data for the current row index
    row_data = data.groupby('File Name').nth(row_index).reset_index()  # Extract the nth row from each group

    # Scatter plot (Center Value vs Integral Value)
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
plt.tight_layout()

# Save the scatter plot
scatter_plot_file = os.path.join(save_path, f"{file_name}_fitting.png")
plt.savefig(scatter_plot_file)

# Box Plot with Average Values
plt.figure(figsize=(10, 6))

# Create a list to store Integral Value for each row index
box_plot_data = []
means = []  # Store mean values for each group
for row_index in range(colornm):
    row_data = data.groupby('File Name').nth(row_index).reset_index()
    values = row_data['Integral Value'].dropna()
    box_plot_data.append(values)
    means.append(values.mean())  # Compute the mean for the current group

# Plot the box plot
boxplot = plt.boxplot(box_plot_data, labels=labels, patch_artist=True, 
                      boxprops=dict(facecolor='skyblue', color='black'),
                      medianprops=dict(color='red'), whiskerprops=dict(color='black'))

# Annotate the mean values above each box
for i, mean in enumerate(means):
    plt.annotate(f'{mean:.2f}',  # Format mean value to 2 decimal places
                 xy=(i + 1, mean),  # (x, y) coordinate
                 xytext=(0, 5),  # Offset text slightly above the mean point
                 textcoords='offset points',
                 ha='center', fontsize=10, color='blue')

# Set axis labels and title for box plot
plt.xlabel('Row Index', fontsize=14)
plt.ylabel('Integral Value', fontsize=14)
plt.title('Distribution of Integral Values by Row Index', fontsize=16)

# Save the box plot
box_plot_file = os.path.join(save_path, f"{file_name}_boxplot_with_means_annotated.png")
plt.savefig(box_plot_file)
