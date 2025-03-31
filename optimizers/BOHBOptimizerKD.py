import os
import shutil
import time
from ray import tune
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter

class BOHBOptimizer:
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        self.config = config
        self.model_wrapper = model_wrapper
        self.model_config = model_config
        self.logging_util = logging_util
        self.seed = seed
        self.best_config = None
        self.best_value = None

    def optimize(self):
        def objective(config, seed, budget):
            config_dict =self.model_config.cs_to_dict(config)
            score = self.model_wrapper.run_model(config_dict, budget)
            #logging does not work for last few iters
            return 1 - score  # SMAC minimizes the objective
        
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
       
        start_time = time.time()
        output_directory = self.config['output_directory']
        cs, _, _ = self.model_config.get_configspace()
        hyperparameter_dict = self.model_config.get_hyperparam_dict()
        max_budget = int(self.model_wrapper.get_train_size())

        # Set up BOHB search algorithm
        bohb_search = TuneBOHB(config_space=config_space, seed=self.seed)
        search_alg = ConcurrencyLimiter(bohb_search, max_concurrent=4)

        # Set up BOHB scheduler
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            metric="loss",
            mode="min",
        )

        # Start logging
        self.logging_util.start_logging()

        # Run the optimization
        analysis = tune.run(
            train_function,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=self.config['n_trials'],
            config={"budget": tune.uniform(self.config['min_budget'], self.config['max_budget'])},
            resources_per_trial={"cpu": 1},
            local_dir=output_directory,
            verbose=1
        )

        # Extract the best configuration and value
        self.best_config = analysis.best_config
        self.best_value = analysis.best_result["loss"]
        print(f"Found best config {self.best_config} with loss: {self.best_value}")

        # Stop logging
        self.logging_util.stop_logging()

    def create_configspace(self, hyperparameter_dict):
        cs = ConfigurationSpace()
        for name, param_list in hyperparameter_dict.items():
            param_list = [round(x, 2) if isinstance(x, float) else x for x in param_list]
            if name == "quantile_subsample":
                param_list = [round(x, 1) if isinstance(x, float) else x for x in param_list]
            hp = CategoricalHyperparameter(name=name, choices=list(set(param_list)))
            cs.add_hyperparameter(hp)
        return cs
