# models/model_wrapper.py
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.test_functions.base_model_func import BaseModelFunc

class OptimizationTestFuncWrapper:
    def __init__(self, model: BaseModelFunc):
        self.model = model

    def run_model(self, hyperparams):
        # Train the model
        value = self.model.evaluate(hyperparams)
        return value
    
    def get_hyperparam_space(self):
        return self.model.get_configspace()
    