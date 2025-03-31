import ast
import csv
import gc
import gzip
import os
import pickletools
import re
import sys
import time
import yaml
from models.DecisionTree import DecisionTree
from models.MultinomialNB import MultinomialNaiveBayes
from models.KNN import KNN
from models.SVM import SVM
from models.LogisticRegression import LogisticRegression
from models.configurations.model_config import ModelConfiguration
from models.model_wrapper import ModelWrapper
from models.random_forest import RandomForest
from optimizers.FAREOptimizerBeta import FAREOptimizerBeta
from optimizers.DefaultOptimizer import DefaultOptimizer
from optimizers.FAREOptimizerEasyExp import FAREOptimizerExp
from optimizers.BOHBOptimizer import BOHBOptimizer
from optimizers.FAREOptimizer import FAREOptimizer
from optimizers.FLASHOptimizer import FLASHOptimizer
from optimizers.GridSearchOptimizer import GridSearchOptimizer
from optimizers.HyperbandOptimizer import HyperbandOptimizer
from optimizers.LineOptimizer import LineOptimizer
from optimizers.RandomSearchOptimizer import RandomSearchOptimizer
from optimizers.TPEOptimizer import TPEOptimizer
from utils.LoggingUtil import LoggingUtil
from utils.clustering_tree_faster import Cluster
from utils.data_loader import load_data, load_data_simple, load_full_data, save_splitted_data
import utils.issue_close_preprocess as issue
import utils.preprocessor as defect
import utils.smells_preprocessor as smell
import pickle
import pandas as pd
import utils.UCI_preprocessor as UCI

seed = None
hyperparameter_configs = {}
pickle_root = "pickles_smote_10"
hyperparameter_configs_key = "hyperparameter_configs"
clusters_key = "hyperparameter_clusters"


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


# Generate train/test file names based on dataset structure
def generate_file_names(dataset):
    train_files = {}
    test_files = {}

    path = dataset['path']
    print(dataset)
    if 'holdout' in dataset and not dataset.get('disable'):
        # Handle holdout datasets
        for name, versions in dataset['holdout']['train'].items():
            train_files.setdefault(name, []).extend(f"{path}/{name}-{version}.csv" for version in versions)
        for name, versions in dataset['holdout']['test'].items():
            test_files.setdefault(name, []).extend(f"{path}/{name}-{version}.csv" for version in versions)
    elif 'cross_validation' in dataset and not dataset.get('disable'):
        # Handle cross-validation datasets
        # Here you would normally read the existing CSV files and perform train-test split
        for name in dataset['cross_validation']['train']:
            #data_name = name.split(dataset['cross_validation']['name_seperator'])[1]
            train_files.setdefault(name, []).append(f"{path}/{name}.csv")
        for name in dataset['cross_validation']['test']:
            #data_name = name.split(dataset['cross_validation']['name_seperator'])[1]
            test_files.setdefault(name, []).append(f"{path}/{name}.csv")
        
        #save_splitted_data(train_files, test_files, dataset['cross_validation']['test_split'])
     
    
    return train_files, test_files

def write_to_file(filepath, content):
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write content to the file
    with open(filepath, 'w') as file:
        file.write(content)

def init_model(model_type, seed=42):
    """Initialize model based on its type and hyperparameters."""
    model_classes = {
        'RandomForest': RandomForest,
        'LogisticRegression': LogisticRegression,
        'KNN': KNN,
        'DecisionTree' : DecisionTree,
        'SVM' : SVM
    }
    if model_type not in model_classes:
        raise ValueError(f"Unknown learner: {model_type}")
    
    return model_classes[model_type](seed)

def init_optimizer(optimizer_name, optimizer_config, model_wrapper, seed=1):
    """Initialize optimizer based on its name and configuration."""
    optimizer_classes = {
        'DEFAULT': DefaultOptimizer,
        'FARE': FAREOptimizerBeta,
        'OPTUNA': TPEOptimizer,
        'Hyperband': HyperbandOptimizer,
        'BOHB': BOHBOptimizer,
        'LINE' : LineOptimizer,
        'FLASH': FLASHOptimizer,
        'RANDOM': RandomSearchOptimizer
    }
    if optimizer_name not in optimizer_classes:
        return None
    
    #initially set model_config to zero, and set after getting seeds
    return optimizer_classes[optimizer_name](optimizer_config, model_wrapper, None, None, seed)

def init_hyperparam_configs(model_name, config, needed):
    model_config = hyperparameter_configs.get(model_name, None)
    if not config:
        model_config = ModelConfiguration(config =config, needed= needed)
        model_config.get_configspace()
        hyperparameter_configs[model_name] = model_config
    return model_config
    
def get_preprocessor(dataset):
    if dataset.get('timed'):
        return defect.preprocess
    if 'holdout' in dataset and not dataset.get('disable') and dataset['task']=='defect prediction':
        return defect.preprocess
    elif 'cross_validation' in dataset and not dataset.get('disable') and dataset['task']=='issue close prediction':
        return issue.preprocess
    
    elif 'cross_validation' in dataset and not dataset.get('disable') and dataset['task']=='code smell detection':
        return smell.preprocess
    
    elif 'cross_validation' in dataset and not dataset.get('disable') and dataset['task']=='non-se task':
        return UCI.preprocess


def dump(key, pickle_object):
    print("DUMP!!!!!!!!")
    os.makedirs(pickle_root, exist_ok=True)
    pickle_path = get_pickle_path(key)
    with open(pickle_path, "wb") as file:
        pickle.dump(pickle_object, file, protocol=pickle.HIGHEST_PROTOCOL)
        return pickle_object

def load(key):
    pickle_path = get_pickle_path(key)    
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
            
    return None  

def get_pickle_path(key):
    return os.path.join(pickle_root, f"{key}.pkl")

def init_experiment(datasets, models, repeats):
    #if os.path.exists(pickle_root) and os.path.isdir(pickle_root):
    #    return
                
    for dataset in datasets:
        if dataset.get('disable'): continue
        train_files, test_files = generate_file_names(dataset)
        class_index = dataset['target_column_index']
        for model_cfg in models:
            model_name = model_cfg['type']
            if model_cfg.get('disable'): continue
            hyperparam_model_key = hyperparameter_configs_key+model_name
            cluster_model_key = clusters_key+model_name
            hyperparam_model_path = get_pickle_path(hyperparam_model_key)
            cluster_model_path = get_pickle_path(cluster_model_key)
            
            for i in range(repeats):
                seed = i+1
                hyperparam_model_key = hyperparameter_configs_key+model_name+str(seed)
                cluster_model_key = clusters_key+model_name+str(seed)
                hyperparam_model_path =  get_pickle_path(hyperparam_model_key)
                cluster_model_path = get_pickle_path(cluster_model_key)
                model_config = None
                if not checkFileExists(hyperparam_model_path):
                    model_config = ModelConfiguration(model_cfg['parameters'], model_cfg.get('conditions'), model_cfg.get('constraints'), seed, model_cfg['needed']) 
                    dump(hyperparam_model_key, model_config)
                if not checkFileExists(cluster_model_path):
                    if not model_config:
                            model_config = load(hyperparam_model_key)
                    cluster = Cluster(model_config, seed)
                    dump(cluster_model_key, cluster)

            for data_name, train_file in train_files.items():
                #assuming the same train file will not be used differently for training
                #retrive pickled instances
                data_file_path = get_pickle_path(data_name)
                if not checkFileExists(data_file_path):
                    if dataset.get('timed'): 
                        if len(train_file) == 1:
                            heldout_train_X, heldout_train_y = None, None
                        else:
                            heldout_train = train_file[-1]
                            heldout_train_X, heldout_train_y = load_data_simple(heldout_train, get_preprocessor(dataset), class_index)
                            train_file = train_file[:-1]
                        X_train, y_train, X_test, y_test = load_data(train_file, test_files[data_name], get_preprocessor(dataset), class_index)
                        dump(data_name, (X_train, y_train, heldout_train_X, heldout_train_y, X_test, y_test))
                    else:
                        X_train, y_train = load_full_data(train_file, get_preprocessor(dataset), class_index)
                        dump(data_name, (X_train, y_train))
            gc.collect()
            
def checkFileExists(hyperparam_model_path):
    return (os.path.exists(hyperparam_model_path) and os.path.isfile(hyperparam_model_path))

def get_avg_configs_up_to_checkpoint(csv_file, checkpoint):
    def get_dict(config_str):
        """Extract content inside the second pair of curly braces and convert it to a dictionary."""
        return ast.literal_eval(config_str)

    with open(csv_file, mode='r') as f:
        data = list(csv.DictReader(f))
       
    # Iterate through each checkpoint and find the best config up to that point
    values = [float(row['value']) for row in data if int(row['iteration']) < checkpoint]
    median_value = sorted(values)[len(values) // 2] if values else float('inf')
    best_config = next((row['config'] for row in data if float(row['value']) == median_value), None)
    best_value = median_value
    elapsed_time = next((float(r['elapsed_time']) for r in data if int(r['iteration']) < checkpoint), None)    
    for r in data:
        if int(r['iteration']) == checkpoint:
            break
        elapsed_time = float(r['elapsed_time']) 
    print(f"Checkpoint {checkpoint}: Best Config: {best_config}, Value: {best_value}")
    return get_dict(best_config), best_value, elapsed_time

def get_best_configs_up_to_checkpoint(csv_file, checkpoint):
    def get_dict(config_str):
        """Extract content inside the second pair of curly braces and convert it to a dictionary."""
        return ast.literal_eval(config_str)

    with open(csv_file, mode='r') as f:
        data = list(csv.DictReader(f))
       
    # Iterate through each checkpoint and find the best config up to that point
    best_config, best_value = min(
        ((row['config'], float(row['value'])) for row in data if int(row['iteration']) < checkpoint),
        key=lambda x: x[1], default=(None, float('inf'), float('inf'))
    )
    elapsed_time = next((float(r['elapsed_time']) for r in data if int(r['iteration']) < checkpoint), None)    
    for r in data:
        if int(r['iteration']) == checkpoint:
            break
        elapsed_time = float(r['elapsed_time']) 
    print(f"Checkpoint {checkpoint}: Best Config: {best_config}, Value: {best_value}")
    return get_dict(best_config), best_value, elapsed_time

def pos_to_neg_ratio(datasets):
    for dataset in datasets:
        if dataset.get('disable'): continue
        train_files, _ = generate_file_names(dataset)
        for data_name, _ in train_files.items():
            X_train, y_train, heldout_train_X, heldout_train_y, X_test, y_test = load(data_name)
            counts = y_train.value_counts()
            ratio = counts.get(1, 0) / counts.get(0, 0) if counts.get(0, 0) > 0 else float('inf')
            print(f"ratio for {data_name} is {ratio}")
        
# Main function to prepare and run optimizers
def run_experiment(datasets, models, optimizers, repeats, checkpoints, tmp_output_dir, logging_dir):
    for dataset in datasets:
        if dataset.get('disable'): continue
        train_files, _ = generate_file_names(dataset)
        
        for model_cfg in models:
            if model_cfg.get('disable'): continue
            model_name = model_cfg['type']
            model = init_model(model_name)
            
             
            for data_name, _ in train_files.items():
                retreived = load(data_name)
                if not retreived:
                    raise ValueError("There was an error in initialization of training files!")
                
                if dataset.get('timed'): 
                    X_train, y_train, heldout_train_X, heldout_train_y, X_test, y_test = retreived
                    if heldout_train_X is not None and heldout_train_y is not None:
                        X_train = pd.concat([X_train, heldout_train_X], axis=0)
                        y_train = pd.concat([y_train, heldout_train_y], axis=0)
                    model_wrapper = ModelWrapper(model, X_train, y_train, None, None, X_test, y_test,)

                else:    
                    X_train, y_train = retreived
                    model_wrapper = ModelWrapper(model, X_train, y_train, None, None)

                for opt_cfg in optimizers:
                    if opt_cfg.get('disable'): continue
                    run_optimizer = True
                    for checkpoint in checkpoints:
                        optimizer_name = opt_cfg['name']
                        results_filepath = os.path.join(tmp_output_dir, optimizer_name, model_cfg['type'],dataset.get('task'), f"{data_name}_{checkpoint}.csv")
                        if checkFileExists(results_filepath):
                            continue
                        if optimizer_name == 'FARE' or optimizer_name == 'BOHB': 
                            opt_cfg['n_trials'] = checkpoint
                        optimizer = init_optimizer(optimizer_name, opt_cfg, model_wrapper)
                        print(f'Running {optimizer_name}')
                        
                        if optimizer:
                            results = {key: [] for key in ["configs", "best_values", "runtimes", "recalls", "false_alarm_revs", "auc", "best_scores"]}
                            
                            for i in range(repeats):
                                hyperconfigs = None
                                clusters = None
                                seed = i+1
                                if optimizer_name == 'RANDOM':
                                    optimizer_log_filename = os.path.join(logging_dir,optimizer_name, model_name, dataset.get('task'),f"{data_name}_{1}.csv")
                                else:
                                    optimizer_log_filename = os.path.join(logging_dir,optimizer_name, model_name, dataset.get('task'),f"{data_name}_{seed}.csv")
                                if run_optimizer:
                                    hyperconfigs = load(hyperparameter_configs_key+model_name+str(seed))
                                    if not hyperconfigs:
                                        raise ValueError(f"Could not load hyperparameter configs for model {model_name}")
                                    optimizer.set_seed(seed)
                                    model.set_seed(seed)
                                    model.reset_model()
                                    optimizer.set_model_config(hyperconfigs)
                                    optimizer.set_logging_util(LoggingUtil(optimizer_log_filename))
                                    if optimizer_name == 'FARE': 
                                        clusters = load(clusters_key+model_name+str(seed))
                                        if not clusters:
                                            raise ValueError(f"Could not load config clusters for model {model_name}")
                                        optimizer.set_cluster(clusters)
                                    
                                    start_time = time.time()
                                    optimizer.optimize()
                                    elapsed_time = time.time() - start_time
                                if optimizer_name == 'FARE' or optimizer_name == 'BOHB' or optimizer_name == 'DEFAULT':
                                    best_config, best_value = optimizer.best_config, (1-optimizer.best_value)
                                else:
                                    if optimizer_name == 'RANDOM':
                                        #best_config, best_value, elapsed_time = get_avg_configs_up_to_checkpoint(optimizer_log_filename, checkpoint)
                                        best_config, best_value, elapsed_time = get_best_configs_up_to_checkpoint(optimizer_log_filename, checkpoint)  
                                    else: 
                                        best_config, best_value, elapsed_time = get_best_configs_up_to_checkpoint(optimizer_log_filename, checkpoint)                                        
                                print(f'Best config for {optimizer_name} on {model_cfg["type"]} for {data_name}: {best_config}, completed in {elapsed_time:.2f}s')
                                
                                # Evaluate and collect results
                                if optimizer_name == 'RANDOM':
                                    recall, false_alarm_rev, auc, comp_score = None, None, None, best_value
                                    metrics = [best_config, best_value, elapsed_time, recall, false_alarm_rev, auc, comp_score]

                                else:
                                    (recall, false_alarm_rev, auc), comp_score = model_wrapper.test(best_config)
                                    metrics = [best_config, best_value, elapsed_time, recall, 1-false_alarm_rev, auc, comp_score]
                                
                                for key, value in zip(results.keys(), map(str, metrics)):
                                    results[key].append(value)
                                    
                            
                            content = '\n'.join([', '.join(results[key]) for key in results])
                            write_to_file(results_filepath, content)
                            if optimizer_name == 'DEFAULT': break
                        if optimizer_name != 'FARE': run_optimizer = False

if __name__ == "__main__":
    config_file = os.path.join("experiments", "final_configs", "config_LINE_Defect.yaml")
    config = load_config(config_file)
    
    init_experiment(config['datasets'], config['models'], config['repeats'])
    run_experiment(config['datasets'], config['models'], config['optimizer'], config['repeats'], config.get('checkpoints'), config['runs_output_folder'], config['logging_folder'])