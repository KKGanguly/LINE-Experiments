from functools import reduce
import itertools
import random

import numpy as np
from sklearn.model_selection import KFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer, MinMaxScaler, MaxAbsScaler, RobustScaler, KernelCenterer, QuantileTransformer, Normalizer
from sklearn.utils import resample
import pandas as pd
from joblib import Parallel, delayed
from random import uniform as _randuniform
from utils.CFS import CFS
class BaseModel:
    def __init__(self,seed):
        self.seed = seed
        self.model = self.create_model()
        random.seed(self.seed)
        
    def set_seed(self, seed):
        self.seed = seed
    
    def reset_model(self):
        self.model = self.create_model()  

    def get_model(self):
        return self.model
    
    def fit(self, X_train, y_train, hyperparameters):
        if hyperparameters: 
            filtered_hyperparameters = {
                k.replace("model_", ""): v for k, v in hyperparameters.items() if k.startswith("model_")
            }   
            self.model.set_params(**filtered_hyperparameters)
            self.model.random_state = self.seed
        self.model.fit(X_train, y_train)
    
      
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
   
    def evaluate(self, X_train, y_train,  X_heldout, y_heldout, X_test, y_test, hyperparameters = None):
        #majority_class_count = y_train.value_counts().max()
        #minority_class_count = y_train.value_counts().min()
        #if minority_class_count<int(0.5*majority_class_count):
        if X_heldout is not None and y_heldout is not None:
            X_train = pd.concat([X_train, X_heldout], ignore_index=True)
            y_train = pd.concat([y_train, y_heldout], ignore_index=True)
            
        X_test_scaled = X_test
        #X_train, y_train, X_test_scaled = self.__select_features__(X_train, y_train, X_test_scaled)
        
        X, y = self.__apply_model__("smote", X_train=X_train, y_train=y_train, hyperparameters=hyperparameters)
        #scaler_names=['standard']
        scaler_names = ['standard', 'minmax', 'maxabs', 'robust', 'quantile', 'normalizer']
        for scaler_name in scaler_names:
            X, X_test_scaled = self.__apply_model__(scaler_name, X_train=X, X_test=X_test_scaled, hyperparameters=hyperparameters)
        self.fit( X, y, hyperparameters)
        y_pred = self.predict(X_test_scaled)
        y_true = np.array(y_test)
        y_preds = np.array(y_pred)
        # update pos_label from config
        # Calculate metrics
        return self.evaluate_prediction(y_true, y_preds)

    def evaluate_prediction(self, y_true, y_preds):
        precision = precision_score(y_true, y_preds, pos_label = 1)
        recall = recall_score(y_true, y_preds, pos_label = 1)

        # Calculate false alarm rate
        tn, fp, _, _ = confusion_matrix(y_true, y_preds).ravel()
        false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        try:
            auc_score = roc_auc_score(y_true, y_preds)
        except ValueError:
            auc_score = 0.0
        #excluding precision
        return (recall, 1-false_alarm, auc_score)
    
    def fast_k_fold(self, kf, X, y, budget, hyperparameters, n_jobs =-1):
        scores = Parallel(n_jobs=n_jobs)(
        delayed(self.run_cross_validation)(train_idx, test_idx, X, y, budget, hyperparameters)
        for train_idx, test_idx in kf.split(X, y)
        )
        return scores
    
    def run_cross_validation(self, train_index, test_index, X_train, y_train, budget, hyperparameters):
        X_train_split, X_test_split = X_train[X_train.index.isin(train_index)], X_train[X_train.index.isin(test_index)]
        y_train_split, y_test_split = y_train[y_train.index.isin(train_index)], y_train[y_train.index.isin(test_index)]
        score = self.evaluate_fold(X_train_split, y_train_split, X_test_split, y_test_split, budget, hyperparameters)
        return score
    
    def run_helf_out_set_validation(self, X_train, y_train, budget, hyperparameters):
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
        score = self.evaluate_fold(X_train_split, y_train_split, X_test_split, y_test_split, budget, hyperparameters)
        return [score]
    
    def cross_validate(self, X_train, y_train, X_holdout, y_holdout, budget = None, hyperparameters = None, n_splits=5, validation_nosmote = False):
        if X_holdout is not None and y_holdout is not None:
            return [self.evaluate_fold(X_train, y_train, X_holdout, y_holdout, budget, hyperparameters)]
        else:
            #kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            #return self.fast_k_fold(kf, X_train, y_train, budget, hyperparameters)
            return self.run_helf_out_set_validation(X_train, y_train, budget, hyperparameters)
        
    def evaluate_fold(self, X_train, y_train, X_holdout, y_holdout, budget, hyperparameters):
        if budget:
            X_train, y_train = resample(
                X_train, y_train, 
                n_samples=int(budget),  # Match original size
                random_state=self.seed,      # Ensure reproducibility
                stratify=y_train         # Maintain class proportions
            )
        X_test_scaled = X_holdout
        #print(X_train)
        #print(X_test_scaled)
        #X_train, y_train, X_test_scaled = self.__select_features__(X_train, y_train, X_test_scaled)
        if hyperparameters:
            X_train, y_train = self.__apply_model__("smote", X_train=X_train, y_train=y_train, hyperparameters=hyperparameters)
        # to ensure class proportion remains same after smoting
            if budget:
                X_train, y_train = resample(
                    X_train, y_train, 
                    n_samples=int(budget),  # Match original size
                    random_state=self.seed,      # Ensure reproducibility
                    stratify=y_train         # Maintain class proportions
                )
        #scaler_names=['standard']
        
        scaler_names = ['standard', 'minmax', 'maxabs', 'robust', 'quantile', 'normalizer']
        for scaler_name in scaler_names:
            X_train, X_test_scaled = self.__apply_model__(scaler_name, X_train=X_train, X_test=X_test_scaled, hyperparameters=hyperparameters)
        self.fit(X_train, y_train, hyperparameters)

        y_preds = self.model.predict(X_test_scaled)
       
        return self.evaluate_prediction(y_holdout, y_preds)
    
    """
    def cross_validate(self, X_train, y_train, X_holdout, y_holdout, budget = None, hyperparameters = None, n_splits=5, validation_nosmote = False):
        if not validation_nosmote:
            classifier_hyperparameters = {f'classifier__{key}': value for key, value in hyperparameters.items()}
            self.pipeline.set_params(**classifier_hyperparameters)
        if X_holdout is None and y_holdout is None:
            X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
        if budget:
            X_train, y_train = resample(
                X_train, y_train, 
                n_samples=int(budget),  # Match original size
                random_state=self.seed,      # Ensure reproducibility
                stratify=y_train         # Maintain class proportions
            )
        X_test_scaled = X_holdout
        #print(X_train)
        #print(X_test_scaled)
        #X_train, y_train, X_test_scaled = self.__select_features__(X_train, y_train, X_test_scaled)
        if hyperparameters:
            X_train, y_train = self.__apply_model__("smote", X_train=X_train, y_train=y_train, hyperparameters=hyperparameters)
        # to ensure class proportion remains same after smoting
            if budget:
                X_train, y_train = resample(
                    X_train, y_train, 
                    n_samples=int(budget),  # Match original size
                    random_state=self.seed,      # Ensure reproducibility
                    stratify=y_train         # Maintain class proportions
                )
        #scaler_names=['standard']
        
        scaler_names = ['standard', 'minmax', 'maxabs', 'robust', 'quantile', 'normalizer']
        for scaler_name in scaler_names:
            X_train, X_test_scaled = self.__apply_model__(scaler_name, X_train=X_train, X_test=X_test_scaled, hyperparameters=hyperparameters)
        self.fit(X_train, y_train, hyperparameters)

        y_preds = self.model.predict(X_test_scaled)
       
        return self.evaluate_prediction(y_holdout, y_preds)
    """
    def __select_features__(self, X_train, y_train, X_test):
        cfs = CFS()
        X_train, y_train = cfs.fit_transform(X_train, y_train)
        X_test = cfs.transform(X_test)
        return X_train, y_train, X_test
    
    def __apply_model__(self, model_name, X_train, X_test = None, hyperparameters = None, y_train=None):
        models = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(),
            'normalizer': Normalizer(),
            'smote': SMOTE(random_state=self.seed)
        }
        if not hyperparameters:
             model = models.get(model_name)
             if model_name == 'smote': 
                 resampled = model.fit_resample(X_train, y_train)
                 return resampled
             else: return model.fit_transform(X_train), model.transform(X_test)
        # Get the model based on the name
        
        model = models.get(model_name)
        if not model: raise ValueError("preprocessor is null")

        # Filter and apply hyperparameters
        filtered_hyperparameters = {k.replace(f"{model_name}_", ""): v 
                                    for k, v in hyperparameters.items() if k.startswith(f"{model_name}_")}
        if 'robust' in model_name:
            # Combine quantile_range_a and quantile_range_b into a tuple for robust scaler
            quantiles = tuple(filtered_hyperparameters.pop(k) for k in ['quantile_range_a', 'quantile_range_b'] if k in filtered_hyperparameters)
            if quantiles: filtered_hyperparameters['quantile_range'] = quantiles

        # Set parameters and transform
        model.set_params(**filtered_hyperparameters)
        
        if model_name == 'smote': 
            model.random_state =self.seed
            while(True):
                try:
                    resampled = model.fit_resample(X_train, y_train)
                    return resampled
                except:
                    model.k_neighbors-=1
                    print("neighbours:", model.k_neighbors)

        else: 
            return model.fit_transform(X_train), model.transform(X_test)
       
