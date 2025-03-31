import os
import csv
import pandas as pd
import shutil

class Result:
    def __init__(self, delta=None, name=None, label_12=None, label_18=None, label_24=None, label_50=None, label_100=None, label_150=None, label_200=None, label_1000=None, file=None, rank_12=None, rank_18=None, rank_24=None, rank_50=None, rank_100=None, rank_150=None, rank_200=None, rank_1000=None, model_name=None):
        self._delta = delta
        self._name = name
        self._label_12 = label_12
        self._label_18 = label_18
        self._label_24 = label_24
        self._label_50 = label_50
        self._label_100 = label_100
        self._label_150 = label_150
        self._label_200 = label_200
        self._label_1000 = label_1000
        self._file = file
        self._rank_12 = rank_12
        self._rank_18 = rank_18
        self._rank_24 = rank_24
        self._rank_50 = rank_50
        self._rank_100 = rank_100
        self._rank_150 = rank_150
        self._rank_200 = rank_200
        self._rank_1000 = rank_1000
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

    def set_label_1000(self, label_1000):
        self._label_1000 = label_1000

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

    def set_rank_1000(self, rank_1000):
        self._rank_1000 = rank_1000

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
            'label_1000': self._label_1000,
            'file': self._file,
            'rank_12': self._rank_12,
            'rank_18': self._rank_18,
            'rank_24': self._rank_24,
            'rank_50': self._rank_50,
            'rank_100': self._rank_100,
            'rank_150': self._rank_150,
            'rank_200': self._rank_200,
            'rank_1000': self._rank_1000,
            'model_name': self._model_name
        }
        
root = 'out_LINE'
tasks = ['defect prediction', 'code smell detection', 'issue lifetime prediction']
models = ['KNN', 'LogisticRegression', 'RandomForest']

datasets = ['DataClass', 'FeatureEnvy', 'GodClass', 'LongMethod', 'SpaghettiCode', 'camel', 'ivy', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xalan', 'xerces', 'chromium', 'eclipse', 'firefox']
bohb_random_forest = []
fare_random_forest = []
default_random_forest = []
optuna_random_forest = []
random_random_forest = []
line_random_forest = []
bohb_logistic_regression = []
fare_logistic_regression = []
default_logistic_regression = []
optuna_logistic_regression = []
random_logistic_regression = []
line_logistic_regression = []
bohb_knn = []
fare_knn = []
default_knn = []
optuna_knn = []
random_knn = []
line_knn = []
for task in tasks:
    for model in models:
        for dataset in datasets:
            path = root + '/' + task + '/' + model + '/' + f'reported_result_{dataset}.csv'
            results = {}
            if os.path.exists(path):
                with open(path, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if row and row[0].startswith('#'):
                            continue
                        if row:
                            name = str(row[1].strip()).split('_')[0]
                            if name == 'asIs':
                                label = '1000'
                                name = 'DEFAULT'
                            else:
                                label = str(row[1].strip()).split('_')[1]
                            result = results.get(name, Result())
                            result.set_name(name)
                            result.set_file(dataset)
                            result.set_model_name(model)
                            print(str(row[1].strip()).split('_'))
                            

                            rank = int(row[0].strip())
                            median_score = float(row[2].strip())
                            if label == '12':
                                result.set_label_12(median_score)
                                result.set_rank_12(rank)
                            elif label == '18':
                                result.set_label_18(median_score)
                                result.set_rank_18(rank)
                            elif label == '24':
                                result.set_label_24(median_score)
                                result.set_rank_24(rank)
                            elif label == '50':
                                result.set_label_50(median_score)
                                result.set_rank_50(rank)
                            elif label == '100':
                                result.set_label_100(median_score)
                                result.set_rank_100(rank)
                            elif label == '150':
                                result.set_label_150(median_score)
                                result.set_rank_150(rank)
                            elif label == '200':
                                result.set_label_200(median_score)
                                result.set_rank_200(rank)
                            elif label == '1000':
                                result.set_label_1000(median_score)
                                result.set_rank_1000(rank)
                            
                            results[name] = result
                            
            for name, result in results.items():            
                if name == 'BOHB' and result._model_name == 'RandomForest':
                    bohb_random_forest.append(result)
                elif name == 'BOHB' and result._model_name == 'LogisticRegression':
                    bohb_logistic_regression.append(result)
                elif name == 'BOHB' and result._model_name == 'KNN':
                    bohb_knn.append(result)
                elif name == 'FARE' and result._model_name == 'RandomForest':
                    fare_random_forest.append(result)
                elif name == 'FARE' and result._model_name == 'LogisticRegression':
                    fare_logistic_regression.append(result)
                elif name == 'FARE' and result._model_name == 'KNN':
                    fare_knn.append(result)
                elif name == 'DEFAULT' and result._model_name == 'RandomForest':
                    default_random_forest.append(result)
                elif name == 'DEFAULT' and result._model_name == 'LogisticRegression':
                    default_logistic_regression.append(result)
                elif name == 'DEFAULT' and result._model_name == 'KNN':
                    default_knn.append(result)
                elif name == 'OPTUNA' and result._model_name == 'RandomForest':
                    optuna_random_forest.append(result)
                elif name == 'OPTUNA' and result._model_name == 'LogisticRegression':
                    optuna_logistic_regression.append(result)
                elif name == 'OPTUNA' and result._model_name == 'KNN':
                    optuna_knn.append(result)
                elif name == 'RANDOM' and result._model_name == 'RandomForest':
                    random_random_forest.append(result)
                elif name == 'RANDOM' and result._model_name == 'LogisticRegression':
                    random_logistic_regression.append(result)
                elif name == 'RANDOM' and result._model_name == 'KNN':
                    random_knn.append(result)
                elif name == 'LINE' and result._model_name == 'RandomForest':
                    line_random_forest.append(result)
                elif name == 'LINE' and result._model_name == 'LogisticRegression':
                    line_logistic_regression.append(result)
                elif name == 'LINE' and result._model_name == 'KNN':
                    line_knn.append(result)
                    
                    
def save_results_to_csv(results, filename):
    data = [result.to_dict() for result in results]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join('rq1_LINE', filename), index=False)
    
if not os.path.exists('rq1_LINE'):
    os.makedirs('rq1_LINE')
else:
    for filename in os.listdir('rq1_LINE'):
        file_path = os.path.join('rq1_LINE', filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            
save_results_to_csv(bohb_random_forest, 'bohb_random_forest_results.csv')
save_results_to_csv(fare_random_forest, 'fare_random_forest_results.csv')
save_results_to_csv(default_random_forest, 'default_random_forest_results.csv')
save_results_to_csv(optuna_random_forest, 'optuna_random_forest_results.csv')
save_results_to_csv(random_random_forest, 'random_random_forest_results.csv')
save_results_to_csv(line_random_forest, 'line_random_forest_results.csv')
save_results_to_csv(bohb_logistic_regression, 'bohb_logistic_regression_results.csv')
save_results_to_csv(fare_logistic_regression, 'fare_logistic_regression_results.csv')
save_results_to_csv(default_logistic_regression, 'default_logistic_regression_results.csv')
save_results_to_csv(optuna_logistic_regression, 'optuna_logistic_regression_results.csv')
save_results_to_csv(random_logistic_regression, 'random_logistic_regression_results.csv')
save_results_to_csv(line_logistic_regression, 'line_logistic_regression_results.csv')
save_results_to_csv(bohb_knn, 'bohb_knn_results.csv')
save_results_to_csv(fare_knn, 'fare_knn_results.csv')
save_results_to_csv(default_knn, 'default_knn_results.csv')
save_results_to_csv(optuna_knn, 'optuna_knn_results.csv')
save_results_to_csv(random_knn, 'random_knn_results.csv')
save_results_to_csv(line_knn, 'line_knn_results.csv')


def print_results(results):
    for result in results:
        print(f'{result._file}--{result._model_name}--')
print("BOHB Logistic Regression Results:")
print_results(bohb_logistic_regression)
print("FARE Logistic Regression Results:")
print_results(fare_logistic_regression)
print("Default Logistic Regression Results:")
print_results(default_logistic_regression)
print("OPTUNA Logistic Regression Results:")
print_results(optuna_logistic_regression)
print("RANDOM Logistic Regression Results:")
print_results(random_logistic_regression)
print("LINE Logistic Regression Results:")
print_results(line_logistic_regression)
print("BOHB KNN Results:")
print_results(bohb_knn)
print("FARE KNN Results:")
print_results(fare_knn)
print("Default KNN Results:")
print_results(default_knn)
print("OPTUNA KNN Results:")
print_results(optuna_knn)
print("RANDOM KNN Results:")
print_results(random_knn)
print("LINE KNN Results:")
print_results(line_knn)