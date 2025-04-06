import pandas as pd
import os
import subprocess

# Read the CSV file
csv_file = pd.read_csv('../baseline.csv')

# Create the output directory if it doesn't exist
output_dir = 'results_LINE_out'
os.makedirs(output_dir, exist_ok=True)

# Traverse the CSV file line by line
output_data = {}

for _, row in csv_file.iterrows():
    file_name = row['File']  # Get the entry from the 'File' column
    for column in ['6', '12', '18', '24', '50', '100']:
        # Create a file name with the column name appended
        output_file = os.path.join(output_dir, f"{file_name}_{column}.csv")
        # Append the corresponding column value to the dictionary
        if output_file not in output_data:
            output_data[output_file] = []
        output_data[output_file].append(str(row[column]))

# Write the constructed strings to their respective files
for output_file, values in output_data.items():
    with open(output_file, 'w') as f:
        #f.write(','.join(values))
        pass

folder = "../results_DEHB/DEHB/moot"

# Get all unique file names in the folder
unique_file_names = set()
for root, _, files in os.walk(folder):
    for file in files:
        # Exclude anything after and including the last underscore
        base_name = file.rsplit('_', 1)[0]
        unique_file_names.add(base_name)

# Print all unique file names
for name in unique_file_names:
    #print(name)
    pass
# Define the folder to traverse
data_folder = "../data/moot"
processed_files = [
    "SS-E", "SS-B", "wc-6d-c1-obj1", "SS-H", "SS-K", "SS-D", "SS-F", "SS-R",
    "sol-6d-c2-obj1", "SS-U", "SS-L", "SS-G", "SS-C", "SS-P", "SS-A", "SS-J",
    "SS-O", "auto93", "SS-S", "SS-Q", "SS-I", "SS-M", "wc+sol-3d-c4-obj1",
    "wc+rs-3d-c4-obj1", "wc+wc-3d-c4-obj1", "SS-V", "SS-T"
]
# Traverse all subfolders inside the data folder
for root, _, files in os.walk(data_folder):
    for file in files:
        print(file)
        # Check if the file matches any name in processed_files with ".csv" appended
        if file in [f"{name}.csv" for name in processed_files]:
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Run the Lua script with the file path as an argument
            for budget in [6,12,18,24,50,100,200]:
                print("results with ", budget, " for ", file_path)
                # Run the Lua script and capture the output
                result = subprocess.run(
                    ["lua5.3", "../kah-main/kah-main/src/kah.lua", "-B", str(budget), "--around", file_path],
                    stdout=subprocess.PIPE,
                    text=True,
                    check=True
                )
                # Process the output to make it CSV-like
                output_lines = result.stdout.strip().split('\n')
                csv_output = ','.join(output_lines)
                
                # Construct the output file path
                output_file_path = os.path.join(
                    "../results_LINE/LINE/moot", f"{os.path.splitext(file)[0]}_{budget}.csv"
                )
                
                # Save the CSV-like output to the file
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, 'w') as output_file:
                    output_file.write(csv_output)



        

