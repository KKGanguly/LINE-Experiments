from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from optimizers.base_optimizer import BaseOptimizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from Nue_main.src.optimization.FlashCV import FlashCV
from sklearn.metrics import make_scorer, roc_auc_score
import joblib
import numpy as np
import pandas as pd

class GridSearchOptimizerFast(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed) -> None:
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.param_names, self.hyperparameter_space = None, None
        self.best_config = None
        self.best_value = None
        
        
    def optimize(self):
        
        # Cache to store previous results
        cache = {}

        def evaluate_params(params):
            # Generate a key for caching
           
            param_key = tuple(sorted(params.items()))
            #if param_key in cache:
            #    return cache[param_key]  # Return cached result
            
            # Set the parameters for the model
            score = self.model_wrapper.run_model(params)
            
            # Cache the result
            #cache[param_key] = score
            print(param_key)
            print(score)
            return score
        _, self.param_names, self.hyperparameter_space = self.model_config.get_configspace()
        param_combinations = [dict(zip(self.param_names, combination)) for combination in self.hyperparameter_space]        # Parallel execution of hyperparameter combinations
        results = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(evaluate_params)(params) for params in param_combinations
        )

        
        # Find the best combination
        best_index = np.argmax(results)
        self.best_params = param_combinations[best_index]
        self.best_score = results[best_index]
        print(self.best_params)
        print(self.best_score)
        
    def save_to_csv(self, filename, best_params, best_score):
        df = pd.DataFrame({
            'params': [best_params],
            'score': [best_score]
        })
        df.to_csv(filename, index=False)    
        
        
        

