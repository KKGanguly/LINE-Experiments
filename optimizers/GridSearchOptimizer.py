from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from optimizers.base_optimizer import BaseOptimizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from Nue_main.src.optimization.FlashCV import FlashCV
from sklearn.metrics import make_scorer, roc_auc_score

class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, logging_util, seed):
        super().__init__(config, model_wrapper, logging_util, seed)
        self.best_config = None
        self.best_value = None
        
        
    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        #params = self.model_wrapper.get_expanded_configs()
        _, keys, combinations = self.model_wrapper.get_configspace()
    
        best_score = -float('inf')
        best_params = None
        self.logging_util.start_logging()
        # Iterate over each combination of parameters
        for index, combo in enumerate(combinations):
            # Create a dictionary of current parameters
            params = dict(zip(keys, combo))
            print(f'Evaluated {index} configurations')
            # Set the parameters for the model
            score = self.model_wrapper.run_model(params)
            self.logging_util.log(params,1-score)
            # Update the best parameters if current score is better
            if score > best_score:
                best_score = score
                best_params = params
        self.best_config = best_params
        self.best_value = best_score
        print(f"Evaluated {len(combinations)} configurations")
        print(f"Found best config {self.best_config} with value: {self.best_value}")
        self.logging_util.stop_logging()
