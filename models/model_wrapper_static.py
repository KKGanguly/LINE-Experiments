# models/model_wrapper.py
import math
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from models.base_model import BaseModel
from utils import DistanceUtil


class ModelWrapperStatic:
    def __init__(self, X, y):
        self.X = X
        self.y = y.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        
    def run_model(self, hyperparams = None, budget=None):
        scores = self.get_score(hyperparams)
        _, avg_d2h = self.__get_avg_d2h__(scores)
        #convert to maximizing (due to a legacy version of this code)
        return 1-avg_d2h

    def get_score(self, hyperparams):
        X, y = self.X, self.y
        scores = None
        if hyperparams:
            # Get the index of the row matching the hyperparams
            hyperparam_values = list(hyperparams.values())
            hyperparam_series = pd.Series(hyperparam_values, index=hyperparams.keys())
            
            # Reorder the dataframe columns to match the order of hyperparams keys
            reordered_X = X[list(hyperparams.keys())]
            index = reordered_X.loc[(reordered_X == hyperparam_series).all(axis=1)].index 
            """
            fr = hyperparam_series.to_frame().T  
            reordered_X.to_csv('reordered_X.csv', index=False)  
            fr.to_csv('hyperparam_series.csv', index=False)   
            merged = pd.concat([reordered_X, y], axis=1)
            merged.to_csv('merged_X_y_recordered.csv', index=False)
            print(index)   
            """
            if not index.empty:
                # Get the corresponding row value from self.y as a tuple
                #1- if minimizing, as it is if maximizing
                scores = tuple(1 - y.loc[index, col].values[0] if col.endswith('-') else y.loc[index, col].values[0] for col in y.columns)
                
            else:
                raise ValueError("No matching hyperparameters found in the dataset.")
        else:
            raise ValueError("No hyperparameters provided.")
        assert scores is not None, "Scores cannot be None"
        return scores
    
    def __get_avg_d2h__(self, scores):
        d2h = DistanceUtil.d2h([1] * len(scores), scores)         
        return scores, d2h
            
    
    #returns score distance, smaller is better
    def evaluate(self, hyperparameters = None):
        if hyperparameters == {}: hyperparameters = None
        scores = self.get_score(hyperparameters)
        avg_metrics, avg_d2h = self.__get_avg_d2h__(scores)
        
        return scores, avg_d2h
    
    def test(self, hyperparameters = None):
        if hyperparameters == {}: hyperparameters = None
        scores = self.get_score(hyperparameters)
        return self.__get_avg_d2h__(scores)
    

