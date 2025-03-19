import csv
from collections import Counter
from pdb import set_trace as st
from matplotlib import pyplot as plt
import os


# File path to your CSV file
wv1=950
# filename = f'{wv1}-{wv1+200}'
path = r'../res/Caf2_03072025_rat/liver_ffpe/HMT_6/result/'
file_path = path + 'detected_peaks.csv'
os.makedirs(path, exist_ok=True)

# Initialize a Counter to count the occurrences of each value
value_counts = Counter()

try:
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        
        for row in reader:
            # Process only non-empty rows
            if row:
                # print(f"Processing row: {row}")  # Debugging: Show each row
                for value in row[1:]:  # Skip the first column (filename)
                    value = value.strip()
                    if value:  # Skip empty cells
                        try:
                            numeric_value = float(value)
                            value_counts[numeric_value] += 1
                        except ValueError:
                            print(f"Skipping invalid value: {value}")  # Handle invalid values gracefully
except FileNotFoundError:
    print(f"Error: File not found at path {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Generate simplified output
if value_counts:
    result = ", ".join([f"{value}:{count}" for value, count in sorted(value_counts.items())])
    # print(f"Value counts (value:count):\n{result}")
else:
    print("No values found or file was empty.")

####save all value as txt file###
with open(path + 'output_counts.txt', 'w') as output_file:
    output_file.write(result)

# Print the top 10 most common values
if value_counts:
    print("Top 20 occurrences:")
    top_10 = value_counts.most_common(20)
    for value, count in top_10:
        print(f"{value}:{count}")
else:
    print("No values found or file was empty.")

############################## change here wavelength range#####################################
wavelength_start = 950
wavelength_end = 1800
###histogram visualization###
# Create a list of all values in the desired range
all_values_in_range = [value for value, count in value_counts.items() for _ in range(count) if wavelength_start <= value <= wavelength_end]

plt.figure(figsize=(10, 6))
plt.hist(all_values_in_range, bins=10, color='coral', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title(f'Value Distribution in Range [{wavelength_start}, {wavelength_end}]')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations for the top 10 most common values
for i, (value, count) in enumerate(top_10):
    plt.annotate(f'{value:.0f}: {count:.0f}', 
                 xy=(value, count), 
                 xytext=(value + 10, count + 5),  # Position the text
                 arrowprops=dict(arrowstyle="->", lw=1.5),
                 fontsize=9, color='blue')

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(path, f'pixel_counting_vis.png'))
