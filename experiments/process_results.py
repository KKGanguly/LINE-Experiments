import argparse
import io
import os
import glob
import shutil
import sys
import pandas as pd
from ezr_24Aug14.stats import SOME, report, return_report

def copy_csv_files(tmp_output_dir, target_dir):
    # Traverse through all the optimizer directories in the tmp_output_dir
    for optimizer_dir in os.listdir(tmp_output_dir):
        optimizer_path = os.path.join(tmp_output_dir, optimizer_dir)
        if os.path.isdir(optimizer_path):
            # Traverse through all model types
            for model_type in os.listdir(optimizer_path):
                model_path = os.path.join(optimizer_path, model_type)
                if os.path.isdir(model_path):
                    # Traverse through all dataset tasks
                    for task in os.listdir(model_path):
                        task_path = os.path.join(model_path, task)
                        if os.path.isdir(task_path):
                            # Find all CSV files in the current task directory
                            for csv_file in glob.glob(os.path.join(task_path, '*.csv')):
                                data_name = os.path.basename(csv_file).split('_')[0]  # Get data_name (e.g., 'ant')
                                # Create target directory structure
                                save_dir = os.path.join(target_dir, model_type, task, data_name)
                                os.makedirs(save_dir, exist_ok=True)
                                # Copy the CSV file to the target directory with the new naming convention
                                new_file_name = f"{optimizer_dir}_{os.path.basename(csv_file)}"
                                shutil.copy(csv_file, os.path.join(save_dir, new_file_name))

def copy_reported_csvs(root_dir, model_list, src_dir):
    # Traverse src_dir and find innermost folders with CSV files
    for dirpath, _, filenames in os.walk(src_dir):
        # Check if there are CSV files starting with "reported_result" in the current directory
        csv_files = [f for f in filenames if f.startswith("reported_result") and f.endswith(".csv")]
        
        if csv_files:
            # Get the immediate parent folder name of the current directory
            parent_folder = os.path.basename(os.path.dirname(dirpath))
            grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))

            # Check if parent_folder matches any model in model_list
            if grandparent_folder in model_list:
                # Define target directory for the model under root_dir
                target_dir = os.path.join(root_dir, parent_folder, grandparent_folder)
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy each matching CSV file to the target directory
                for csv_file in csv_files:
                    src_file = os.path.join(dirpath, csv_file)
                    dest_file = os.path.join(target_dir, csv_file)
                    shutil.copy(src_file, dest_file)
                    print(f"Copied {csv_file} to {target_dir}")

def process_csv_files_in_all_inner_folders(folder_path):
    def extract_and_join(s):
        return f"{s.split('_')[0]}_{s.rsplit('_', 1)[-1]}"
    # Walk through all directories starting from the given folder
    average_recall = {}
    average_false_alarm = {}
    average_AUC = {}
    for root, dirs, files in os.walk(folder_path):
        
        if dirs:
            continue
        default = False
        # Check if there are any CSV files in the current directory
        csv_files = [f for f in files if f.endswith('.csv')]

        somes = []
        
        if csv_files:
            # Process each CSV file in the current directory
            for csv_file in csv_files:
                if csv_file.startswith("Default") and default:
                    continue
                csv_path = os.path.join(root, csv_file)
                # Read the CSV file and get the last line
                #print(csv_path)
                df = pd.read_csv(csv_path,skiprows=1)
                print(csv_path)
                last_line = df.iloc[-1]
                last_line_values = last_line.values.tolist()  # Extract as a list
                # Prepare the file base name
                file_base_name = os.path.splitext(csv_file)[0]  # Remove .csv
                name = file_base_name.rsplit('_', 1)[0]
                #print(df.iloc[2].to_list())
                average_recall.setdefault(name, []).extend(df.iloc[2].to_list())
                average_false_alarm.setdefault(name, []).extend(df.iloc[3].to_list())
                average_AUC.setdefault(name, []).extend(df.iloc[4].to_list())
                #file_base_name = "_".join(file_base_name.rsplit('_', 1))  # Remove last underscore part
                file_base_name = extract_and_join(file_base_name)
                # Create the SOME object
                some_value = last_line_values  # Concatenate values and filename
                print(some_value)
                third_line = df.iloc[2]  # Get the third line (index 2)
                #print(file_base_name)
                #print(third_line)
                third_line_values = third_line.values.tolist() 
                # Calculate average time from the last line (assuming last line is the performance metrics)
                #average_time = sum(third_line_values) / len(third_line_values) if last_line_values else 0
                
                #print(f"{name.split("_")[1]},{file_base_name.split("_")[0]},{average_time}")
                # Append to somes with a descriptive label
               
                
                if csv_file.startswith("Default"):
                    name = "asIs"
                    default = True
                else:
                    name = file_base_name
                #print(name)
                some_value = [float(val) for val in some_value]
                somes.append(SOME(some_value, f"{name}"))
        # Report the results
        reported_result = capture_print_output(report,somes, 0.01)
        #print(os.path.basename(root))
        report_file_path = os.path.join(root, f'reported_result_{os.path.basename(root)}.csv')
        with open(report_file_path, 'w') as report_file:
            # Write header
            report_file.write(reported_result)
    """
    print("recall")
    for key, values in average_recall.items():
        print(key, sum(values)/len(values))
    print("False Alarm")
    for key, values in average_false_alarm.items():
        print(key, sum(values)/len(values))
    print("AUC")
    for key, values in average_AUC.items():
        print(key, sum(values)/len(values))
    """
   
def capture_print_output(func, *args, **kwargs):
    # Create a StringIO object to capture the output
    captured_output = io.StringIO()
    # Redirect stdout to the StringIO object
    sys.stdout = captured_output
    try:
        # Call the function with any arguments it requires
        func(*args, **kwargs)
    finally:
        # Reset stdout to its original state
        sys.stdout = sys.__stdout__
    # Get the output and return it
    return captured_output.getvalue()
    
def main(input_dir, results_ind_dir = 'results_tmp', final_results_dir= 'final_results'):
    os.makedirs(results_ind_dir, exist_ok=True)

    # Copy CSV files
    copy_csv_files(input_dir, results_ind_dir)

    # Process CSV files
    process_csv_files_in_all_inner_folders(results_ind_dir)

    # Copy reported CSVs
    root_dir = results_ind_dir
    model_list = ['RandomForest', 'LogisticRegression', 'KNN']
    copy_reported_csvs(final_results_dir, model_list, root_dir)

    # Remove temporary directory
    shutil.rmtree(results_ind_dir)

    """
    tmp_output_dir = os.path.join("res2","runs")  # Replace with your actual tmp output directory
target_dir = "results_ind"
shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)
copy_csv_files(tmp_output_dir, target_dir)
process_csv_files_in_all_inner_folders('results_ind')
root_dir = 'results_ind'
model_list = ['DecisionTree','RandomForest', 'LogisticRegression', 'KNN' ]
copy_reported_csvs('final_results_2', model_list, root_dir)

    """
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file paths as command-line arguments.")
    parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing run files.")
    parser.add_argument('--final_results_dir', type=str, required=True, help="Directory to save final reported results.")

    args = parser.parse_args()

    main(input_dir=args.input_dir, final_results_dir=args.final_results_dir)

