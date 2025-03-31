import ast
import os
import shutil
import time
from optimizers.base_optimizer import BaseOptimizer
from smac import Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from smac import MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband
from ConfigSpace.configuration import Configuration
import ConfigSpace as CS
import random
class CustomConfigurationSpace(CS.ConfigurationSpace):
    def __init__(self, predefined_configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predefined_configs = predefined_configs
    def sample_configuration(self, size=1):
        # Sample configurations from the predefined list
        #print("TRACK=>", self.track)
        if size == 1:
            sampled_dict = random.sample(self.predefined_configs, size)
            sample = sampled_dict[0]
            
            if "model_penalty" in sample and "model_solver" in sample:
                model_penalty = sample["model_penalty"]
                if model_penalty == 'l1' and sample["model_solver"] not in ['liblinear', 'saga']:
                    # If model_penalty is 'l1', and model_solver is invalid, suggest again
                    sample["model_solver"] = random.choice(['liblinear', 'saga'])
                elif model_penalty == 'l2' and sample["model_solver"] not in ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
                    # If model_penalty is 'l2', and model_solver is invalid, suggest again
                    sample["model_solver"] = random.choice(['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
            with open("samples.txt", "a") as f:
                f.write(f'{sample}\n')
            return Configuration(self, values=sample)
        else:
            sample_dicts = random.sample(self.predefined_configs, size)
            samples=[]
            for sample in sample_dicts:
                with open("samples.txt", "a") as f:
                    f.write(f'{sample}\n')
                if "model_penalty" in sample and "model_solver" in sample:
                    model_penalty = sample["model_penalty"]
                    if model_penalty == 'l1' and sample["model_solver"] not in ['liblinear', 'saga']:
                        # If model_penalty is 'l1', and model_solver is invalid, suggest again
                        sample["model_solver"] = random.choice(['liblinear', 'saga'])
                    elif model_penalty == 'l2' and sample["model_solver"] not in ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
                        # If model_penalty is 'l2', and model_solver is invalid, suggest again
                        sample["model_solver"] = random.choice(['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
                    samples.append(Configuration(self, values=sample))
            return samples
        
class BOHBOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
            
    def optimize(self):
        
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        
        def objective(config, seed, budget):
            config_dict =self.model_config.cs_to_dict(config)
            score = self.model_wrapper.run_model(config_dict, budget)
            #logging does not work for last few iters
            self.logging_util.log(config_dict, (1-score), (time.time()-start_time))
            return 1 - score  # SMAC minimizes the objective
        start_time = time.time()
        output_directory = self.config['output_directory']
        random.seed(self.seed)
        #hyperparameter_dict = self.model_config.get_hyperparam_dict()
        max_budget = int(self.model_wrapper.get_train_size())
        cs = self.create_configspace(self.model_config)
        scenario = Scenario(
            cs,
            min_budget=self.config['min_budget'],  # Min budget (in epochs or time)
            output_directory = output_directory,
            max_budget=max_budget,  # Max budget (in epochs or time)
            n_trials = self.config['n_trials'],
            seed = self.seed
        )
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        print(self.config['initial_configs'])
        # Initialize the HBFacade (Hyperband) optimizer with the objective function
        initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=self.config['initial_configs'])
        intensifier = Hyperband(scenario, seed = self.seed)
        optimizer = MultiFidelityFacade(scenario, initial_design = initial_design, target_function=objective, intensifier=intensifier)
        self.logging_util.start_logging()
        
        # Run the optimizer
        incumbent = optimizer.optimize()
        total_evaluations = len(optimizer.runhistory)
        self.best_config = incumbent.get_dictionary()
        self.best_value = objective(incumbent, 0, 0)
        print(f"Evaluated {total_evaluations} configurations")
        print(f"Found best config {self.best_config} with value: {1-self.best_value}")
        self.logging_util.stop_logging()
    
    def create_configspace(self, model_config):
        config_space, param_names, space = self.model_config.get_configspace()
        config_dict = [dict(zip(param_names, values)) for values in space]
        random.shuffle(config_dict)
        #to reduce BOHB runtime, we take 20000 samples, this has no impact over accuracy as labels <=50
        config_dict = config_dict[:20000]
        cs = CustomConfigurationSpace(config_dict)
        for hyperparameter in config_space.get_hyperparameters():
            cs.add_hyperparameter(hyperparameter)
        # Convert each parameter to a CategoricalHyperparameter
       
        return config_space