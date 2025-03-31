import os
import shutil
import time
from optimizers.base_optimizer import BaseOptimizer
from smac import HyperparameterOptimizationFacade, Scenario
from smac import HyperbandFacade as HBFacade
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter

class HyperbandOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        
        def objective(config, seed, budget):
            score = self.model_wrapper.run_model(config, budget)
            config_dict =self.model_config.cs_to_dict(config)
            self.logging_util.log(config_dict, 1-score, (time.time() - start))
            return 1 - score  # SMAC minimizes the objective
        
        output_directory = self.config['output_directory']
        cs, _, _ = self.model_config.get_configspace()
        
        scenario = Scenario(
            cs,
            min_budget=self.config['min_budget'],  # Min budget (in epochs or time)
            output_directory = output_directory,
            max_budget=self.config['max_budget'],  # Max budget (in epochs or time)
            n_trials = self.config['n_trials'],
            seed = self.seed
        )
        
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
            
        # Initialize the HBFacade (Hyperband) optimizer with the objective function
        optimizer = HBFacade(scenario, target_function=objective)
        # Run the optimizer
        self.logging_util.start_logging()
        start = time.time()
        incumbent = optimizer.optimize()
        total_evaluations = len(optimizer.runhistory)
        self.best_config = incumbent.get_dictionary()
        self.best_value = objective(incumbent, 0, 0)
        print(f"Evaluated {total_evaluations} configurations")
        print(f"Found best config {self.best_config} with value: {1-self.best_value}")
        self.logging_util.stop_logging()
        

    