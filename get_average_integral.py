import pandas as pd
import glob
import os
from pdb import set_trace as st


foldername_list = ['1000/','9010/','8020/','7030/','6040/']
path = '../res/Caf2_09022025_amide1/second_derivative/'
for file in foldername_list:
    # Set your CSV directory
    folder_path = path+file+'/subspectrum_fitting_results.csv'  # <- change this to your actual folder
    # Load the CSV
    df = pd.read_csv(folder_path)  # Replace with your actual file name

    # Add a line number within each file group
    df['Peak Index'] = df.groupby('File Name').cumcount()

    # Group by Peak Index to get mean and std across different File Names
    result = df.groupby('Peak Index')['Integral Value'].agg(['mean', 'std']).reset_index()

    # Print or save
    print(result)
    # st()
    # Optional: Save to CSV
    result.to_csv(path+file+'/integral_value_summary.csv', index=False)

