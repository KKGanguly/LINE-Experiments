import csv
import os
import pandas as pd 
# Function to extract the second line (runtime values) from a CSV file
def extract_second_line(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        
        # Read all rows and return the second row
        rows = list(reader)
        
        # Assuming the second line contains the runtime values
        second_line = rows[1]  # Index 1 corresponds to the second row
        return [float(value) for value in second_line]

# Function to process the files in the folder and create a dictionary
def process_folder(folder_path, runtime_dict):
    
    
    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)
            
            # Extract dataset name before the underscore
            dataset_name = filename.split('_')[0]
            
            # Extract runtime values
            runtime_values = extract_second_line(file_path)
            
            # Add runtime values to the dictionary
            if dataset_name not in runtime_dict:
                runtime_dict[dataset_name] = []
            runtime_dict[dataset_name].extend(runtime_values)
    
    return runtime_dict

# Example usage
runtime_dict = {}
folder_path = 'res2/runs/OPTUNA/KNN/code smell detection/'  # Replace with the path to your folder
runtime_data = process_folder(folder_path, runtime_dict)
folder_path = 'res2/runs/OPTUNA/RandomForest/code smell detection/'  # Replace with the path to your folder
runtime_data = process_folder(folder_path, runtime_data)
folder_path = 'res2/runs/OPTUNA/LogisticRegression/code smell detection/'  # Replace with the path to your folder
runtime_data = process_folder(folder_path, runtime_data)
df = pd.DataFrame(runtime_data)
df.to_csv('OPTUNA_codesmell.csv', index=False)

# Example usage
#csv_file = 'results_FARE_Complete_Final_Results/FARE/KNN/code smell detection/DataClass_12.csv'  # Replace with the path to your CSV file
#runtime_values = extract_second_line(csv_file)
#print(runtime_values)