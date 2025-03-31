import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_simple(file, preprocessor, class_index = -1):
    data = pd.read_csv(file)
    X_data, y_data = data.iloc[:, :class_index], data.iloc[:, class_index]
    return preprocessor(X_data,y_data)

def load_data(train_files, test_files, preprocessor, class_index = -1):
    """Load train and test data from CSV files."""
    train_data_list = [pd.read_csv(file) for file in train_files]
    train_data = pd.concat(train_data_list, ignore_index=True)
    test_data_list = [pd.read_csv(file) for file in test_files]
    test_data = pd.concat(test_data_list, ignore_index=True)
    
    X_train, X_test = train_data.iloc[:, :class_index], test_data.iloc[:, :class_index]
    y_train, y_test = train_data.iloc[:, class_index], test_data.iloc[:, class_index]
    file = train_files[0].lower()
    if "issue" in file:
        X_train, y_train, X_test, y_test = preprocessor(X_train, y_train, X_test, y_test, file)
    
    else: 
        X_train, y_train = preprocessor(X_train, y_train)
        X_test, y_test = preprocessor(X_test, y_test)
    return X_train, y_train, X_test, y_test

def load_full_data(train_files, preprocessor, class_index = -1):
    """Load train and test data from CSV files."""
    train_data_list = [pd.read_csv(file) for file in train_files]
    train_data = pd.concat(train_data_list, ignore_index=True)
    
    X_train = train_data.iloc[:, :class_index]
    y_train = train_data.iloc[:, class_index]
    file = train_files[0].lower()
   
    if "issue" in file:
        X_train, y_train, _, _  = preprocessor(X_train, y_train, None, None, file)
    
    else: 
        X_train, y_train = preprocessor(X_train, y_train)
        
    return X_train, y_train

def save_splitted_data(train_files, test_files, test_frac):
    def get_path(data,name):
        return os.path.join(os.path.dirname(data),f"{name}.csv")
    train_data = {name: pd.read_csv(get_path(filepath[0], name)) for name,filepath in train_files.items()}
    for name, train_data in train_data.items():
        if file_exists(train_files[name][0]) and file_exists(test_files[name][0]):
            continue
        
        train_df, test_df = train_test_split(train_data, test_size=test_frac, random_state=42)

        # Write each to a CSV file
        train_df.to_csv(train_files[name][0], index=False)
        test_df.to_csv(test_files[name][0], index=False)

def file_exists(name):
    return os.path.exists(name) and os.path.isfile(name)

