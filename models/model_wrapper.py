# models/model_wrapper.py
import math
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.base_model import BaseModel
from utils import DistanceUtil


class ModelWrapper:
    def __init__(self, model:BaseModel, X_train, y_train, X_test, y_test, X_heldout=None, y_heldout=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_heldout = X_heldout
        self.y_heldout = y_heldout
        
    def run_model(self, hyperparams = None, budget=None):
        # Perform cross-validation and return the mean score
        #start = time.time()
        train_X, train_y = self.X_train, self.y_train
        #if budget: train_X, train_y = self.__sample_aligned_data(self.X_train, self.y_train, budget)
        scores = self.model.cross_validate(train_X, train_y, self.X_heldout, self.y_heldout, budget, hyperparams, validation_nosmote = True)
        #print("model running time:", time.time() - start)
        avg_metrics, avg_d2h = self.__get_avg_d2h__(scores)
        return 1-avg_d2h
    
    def __get_avg_d2h__(self, scores):
        d2h_values = [DistanceUtil.d2h([1] * len(score), score) for score in scores]
        avg_d2h = sum(d2h_values) / len(scores)

        recalls, rev_false_alarms, auc_scores = zip(*scores)  
        avg_metrics = (sum(recalls) / len(scores), 
                    sum(rev_false_alarms) / len(scores), 
                    sum(auc_scores) / len(scores))
        
        return avg_metrics, avg_d2h
            
    def get_data_percentage(self,size):
        return math.ceil((size/len(self.X_train))*100)
    
    def __sample_aligned_data(self, data1, data2, budget, per_instance_cost = 1):
        n_rows = min(len(data1), math.ceil(budget*per_instance_cost))
        random_indices = data1.sample(n_rows, random_state = self.model.seed).index
        sampled_data1 = data1.loc[random_indices]
        sampled_data2 = data2[random_indices]
        return sampled_data1, sampled_data2
    
    #returns score distance, smaller is better
    def evaluate(self, hyperparameters = None):
        if hyperparameters == {}: hyperparameters = None
        scores = self.model.evaluate(self.X_train, self.y_train,  self.X_heldout, self.y_heldout, self.X_test, self.y_test, hyperparameters)
        return scores, DistanceUtil.d2h([1]*len(scores), scores)
    
    def test(self, hyperparameters = None):
        if hyperparameters == {}: hyperparameters = None
        scores = self.model.cross_validate(self.X_train, self.y_train,  self.X_heldout, self.y_heldout, hyperparameters=hyperparameters)
        return self.__get_avg_d2h__(scores)
    
    def evaluate_predictions(self, y_true, y_preds):
        scores = self.model.evaluate_prediction(y_true, y_preds)
        return 1-DistanceUtil.d2h([1]*len(scores), scores)
    
    def get_train_size(self):
        return len(self.X_train)
    
    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def get_model(self):
        return self.model.get_model()


