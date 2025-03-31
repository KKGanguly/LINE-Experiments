import pandas as pd
import os

folder_path = '../data/UCI'  # Replace with the path to your folder

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize a set to store unique values
unique_values = set()

# Iterate over each CSV file
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Get the unique values from the last column
    last_column = df.columns[-1]
    unique_values.update(df[last_column].unique())

# Print the unique values
print(unique_values)