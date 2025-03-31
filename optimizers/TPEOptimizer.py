import math
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from optimizers.base_optimizer import BaseOptimizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler

class TPEOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
        
    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        
        def objective(trial):
            # Get dynamically suggested hyperparameters
            suggested_params = self.suggest_params(trial, hyperparameter_dict_unique_values)
            score = self.model_wrapper.run_model(suggested_params)
            self.logging_util.log(suggested_params, 1-score, (time.time() - start))
            return score
        
        hyperparameter_dict = self.model_config.get_hyperparam_dict()
        hyperparameter_dict_unique_values = {}
        for name, param_list in hyperparameter_dict.items():
            hyperparameter_dict_unique_values[name] = set(param_list)
        study = optuna.create_study(direction=self.config['direction'],sampler = TPESampler(seed=self.seed))
        self.logging_util.start_logging()
        start = time.time()
        study.optimize(objective, n_trials=self.config['n_trials'])
        self.best_config = study.best_trial.params
        self.best_value = study.best_trial.value 
        print(f"Found best config {self.best_config} with value: {self.best_value}")
        self.logging_util.stop_logging()
        
    def suggest_params(self, trial, hyperparameter_dict):
        suggested_params = {}
        
        # Loop through each parameter and corresponding hyperparameter space
        for name, param_list in hyperparameter_dict.items():
            # Determine the type of the parameter from the first value in the list of values
            suggested_params[name] = trial.suggest_categorical(name, param_list)
            
        if "model_penalty" in suggested_params and "model_solver" in suggested_params:
            model_penalty = suggested_params["model_penalty"]
            if model_penalty == 'l1' and suggested_params["model_solver"] not in ['liblinear', 'saga']:
                # If model_penalty is 'l1', and model_solver is invalid, suggest again
                suggested_params["model_solver"] = random.choice(['liblinear', 'saga'])
            elif model_penalty == 'l2' and suggested_params["model_solver"] not in ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
                # If model_penalty is 'l2', and model_solver is invalid, suggest again
                suggested_params["model_solver"] = random.choice(['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
        
        return suggested_params

    