import csv
import re
import os

def process_csv(input_csv, output_csv, output_folder = 'RQ1_Results'):
    results = []

    with open(input_csv, 'r') as file:
        lines = file.readlines()

    dataset_name = os.path.basename(input_csv)
    current_rank = None
    rank_zero_optimizers = []

    for line in lines:
        line = line.strip()

        if line.startswith("#"):
            # Rank header line
            if current_rank == 0 and rank_zero_optimizers:
                # Process rank zero optimizers
                selected_optimizer = None

                # Filter FARE optimizers
                fare_optimizers = [opt for opt in rank_zero_optimizers if "FARE" in opt["name"]]

                if fare_optimizers:
                    # Pick the FARE optimizer with the lowest evaluations
                    selected_optimizer = min(fare_optimizers, key=lambda x: x["evaluations"])
                else:
                    # Pick the optimizer with the lowest evaluations
                    selected_optimizer = min(rank_zero_optimizers, key=lambda x: x["evaluations"])

                results.append([dataset_name, selected_optimizer["name"], selected_optimizer["median"]])

            # Reset for the next rank
            current_rank = None
            rank_zero_optimizers = []
            continue

        if line:
            # Parse the CSV line
            parts = [p.strip() for p in line.split(",")]
            rank = int(parts[0])
            optimizer = parts[1]
            median_performance = float(parts[2])

            # Extract the number of evaluations from the optimizer name
            match = re.search(r"_(\d+)$", optimizer)
            evaluations = int(match.group(1)) if match else float('inf')

            if rank == 0:
                current_rank = 0
                rank_zero_optimizers.append({
                    "name": optimizer,
                    "median": median_performance,
                    "evaluations": evaluations
                })

    # Handle the last rank
    if current_rank == 0 and rank_zero_optimizers:
        selected_optimizer = None

        # Filter FARE optimizers
        fare_optimizers = [opt for opt in rank_zero_optimizers if "FARE" in opt["name"]]

        if fare_optimizers:
            # Pick the FARE optimizer with the lowest evaluations
            selected_optimizer = min(fare_optimizers, key=lambda x: x["evaluations"])
        else:
            # Pick the optimizer with the lowest evaluations
            selected_optimizer = min(rank_zero_optimizers, key=lambda x: x["evaluations"])

        results.append([dataset_name, selected_optimizer["name"], selected_optimizer["median"]])
    # Write the results to a CSV file
    with open(output_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        if os.stat(output_csv).st_size == 0:  # Check if the file is empty
            writer.writerow(["Dataset", "Optimizer", "Median Performance"])  # Write header only once
        writer.writerows(results)

def process_folder(input_folder, output_csv, output_folder):
    # Process all CSV files in the folder
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv = os.path.join(input_folder, filename)
            process_csv(input_csv, output_csv, output_folder)

# Example usage
tasks = ['code smell detection', 'defect prediction', 'issue lifetime prediction']
models = ['KNN', 'LogisticRegression','RandomForest']
output_folder = 'RQ1_Results'

for task in tasks:
    for model in models:
        input_folder = f'final_results/{task}/{model}/'
        if os.path.isdir(input_folder):    
            output_csv = f'{output_folder}/output_{task}_{model}.csv'
            process_folder(input_folder, output_csv,output_folder)

# Example usage
#process_csv('final_results/defect prediction/RandomForest/reported_result_synapse.csv', 'output_camel.csv')
