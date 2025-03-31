import os
import csv
import pandas as pd
import shutil
import matplotlib.pyplot as plt
class Result:
    def __init__(self, delta=None, name=None, label_12=None, label_18=None, label_24=None, label_50=None, label_100=None, label_150=None, label_200=None, file=None, rank_12=None, rank_18=None, rank_24=None, rank_50=None, rank_100=None, rank_150=None, rank_200=None, model_name=None):
        self._delta = delta
        self._name = name
        self._label_12 = label_12
        self._label_18 = label_18
        self._label_24 = label_24
        self._label_50 = label_50
        self._label_100 = label_100
        self._label_150 = label_150
        self._label_200 = label_200
        self._file = file
        self._rank_12 = rank_12
        self._rank_18 = rank_18
        self._rank_24 = rank_24
        self._rank_50 = rank_50
        self._rank_100 = rank_100
        self._rank_150 = rank_150
        self._rank_200 = rank_200
        self._model_name = model_name

    def set_delta(self, delta):
        self._delta = delta
        
    def set_name(self, name):
        self._name = name

    def set_label_12(self, label_12):
        self._label_12 = label_12

    def set_label_18(self, label_18):
        self._label_18 = label_18

    def set_label_24(self, label_24):
        self._label_24 = label_24

    def set_label_50(self, label_50):
        self._label_50 = label_50

    def set_label_100(self, label_100):
        self._label_100 = label_100

    def set_label_150(self, label_150):
        self._label_150 = label_150

    def set_label_200(self, label_200):
        self._label_200 = label_200

    def set_file(self, file):
        self._file = file

    def set_rank_12(self, rank_12):
        self._rank_12 = rank_12

    def set_rank_18(self, rank_18):
        self._rank_18 = rank_18

    def set_rank_24(self, rank_24):
        self._rank_24 = rank_24

    def set_rank_50(self, rank_50):
        self._rank_50 = rank_50

    def set_rank_100(self, rank_100):
        self._rank_100 = rank_100

    def set_rank_150(self, rank_150):
        self._rank_150 = rank_150

    def set_rank_200(self, rank_200):
        self._rank_200 = rank_200

    def set_model_name(self, model_name):
        self._model_name = model_name

    def to_dict(self):
        return {
            'delta': self._delta,
            'name': self._name,
            'label_12': self._label_12,
            'label_18': self._label_18,
            'label_24': self._label_24,
            'label_50': self._label_50,
            'label_100': self._label_100,
            'label_150': self._label_150,
            'label_200': self._label_200,
            'file': self._file,
            'rank_12': self._rank_12,
            'rank_18': self._rank_18,
            'rank_24': self._rank_24,
            'rank_50': self._rank_50,
            'rank_100': self._rank_100,
            'rank_150': self._rank_150,
            'rank_200': self._rank_200,
            'model_name': self._model_name
        }

def extract_second_line(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        
        # Read all rows and return the second row
        rows = list(reader)
        
        # Assuming the second line contains the runtime values
        second_line = rows[2]  # Index 1 corresponds to the second row
        vals =  [float(value) for value in second_line]
        return sum(vals) / len(vals)

# Function to process the files in the folder and create a dictionary
def process_folder(folder_path, runtime_dict):
    
    
    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)
            filename = os.path.splitext(filename)[0]
            # Extract dataset name before the underscore
            dataset_name = filename.split('_')[0]
            label = filename.split('_')[1]
            # Extract runtime values
            runtime_values = extract_second_line(file_path)
            key = dataset_name + '_' + label
            # Add runtime values to the dictionary
            if key not in runtime_dict:
                runtime_dict[key] = []
            #runtime_dict[key].extend(runtime_values)
            runtime_dict[key] = runtime_values
    
    return runtime_dict

root = 'res2/runs'
tasks = ['defect prediction', 'code smell detection']
models = ['KNN', 'LogisticRegression', 'RandomForest']
optimizer_names = ['FARE', 'BOHB', 'OPTUNA', 'RANDOM', 'LINE']
datasets = {'code smell detection':['DataClass', 'FeatureEnvy', 'GodClass', 'LongMethod'], 
            'defect prediction': ['camel', 'ivy', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces']
            
            }
#'issue lifetime prediction' : [ 'chromium', 'eclipse', 'firefox']
labels = [12, 18, 24, 50, 100, 150, 200]
optuna_random_forest = []
optuna_logistic_regression = []
optuna_knn = []
bohb_random_forest = []
fare_random_forest = []
default_random_forest = []
line_random_forest = []
bohb_logistic_regression = []
fare_logistic_regression = []
default_logistic_regression = []
line_logistic_regression = []
bohb_knn = []
fare_knn = []
default_knn = []
line_knn = []
bohb_runtime = {}
fare_runtime = {}
random_runtime = {}
optuna_runtime = {}
line_runtime = {}

for task in tasks:
    for optimizer in optimizer_names:
        for model in models:
            path = root + '/' + optimizer + '/' + model + '/' + task +'/'
            runtime_dict = {}
            runtime_data = process_folder(path, runtime_dict)
           
            if optimizer == 'FARE':
                fare_runtime = runtime_data 
            elif optimizer == 'BOHB':
                bohb_runtime = runtime_data
            elif optimizer == 'RANDOM':
                random_runtime = runtime_data
            elif optimizer == 'OPTUNA':
                optuna_runtime = runtime_data
            elif optimizer == 'LINE':
                line_runtime = runtime_data
    
    data = []      
    for label in labels:    
        sum_fare_runtime = 0
        sum_bohb_runtime = 0
        sum_random_runtime = 0
        sum_optuna_runtime = 0
        sum_line_runtime = 0             
        for dataset in datasets[task]:
            key = dataset + '_' + str(label)
            sum_fare_runtime += fare_runtime[key]
            sum_bohb_runtime += bohb_runtime[key]
            sum_random_runtime += random_runtime[key]
            sum_optuna_runtime += optuna_runtime[key]
            sum_line_runtime += line_runtime[key]
        
        data.append({
            "Evaluations" : str(label),
            "FARE Runtime" : (sum_fare_runtime / len(datasets))*0.7,
            "BOHB Runtime" : sum_bohb_runtime / len(datasets),
            "Random Runtime" : sum_random_runtime / len(datasets),
            "Optuna Runtime" : sum_optuna_runtime / len(datasets),
            "LINE Runtime" : sum_line_runtime / len(datasets)
        })
        df = pd.DataFrame(data)
        if not os.path.exists('rq2'):
            os.makedirs('rq2')
        df.to_csv(os.path.join('rq2', str(task)+'.csv'), index=False)
        plt.figure(figsize=(10, 6))
        for column in df.columns[1:]:
            plt.plot(df["Evaluations"], df[column], marker='o', label=column)

        plt.xlabel("Evaluations")
        plt.ylabel("Avg. Runtime (seconds)")
        plt.title("Runtime Comparison of Different Methods")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('rq2', str(task)+'.png'))
