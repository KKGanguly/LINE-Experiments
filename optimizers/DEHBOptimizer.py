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
from dehb import DEHB
class CustomConfigurationSpace(ConfigurationSpace):
    def __init__(self, predefined_configs, mapping = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predefined_configs = predefined_configs
        self.mapping = mapping
    
    def set_mapping(self, mapping):
        self.mapping = mapping
    
    def get_mapping(self):
        if self.mapping is None:
            return None
        return self.mapping
    
    def sample_configuration(self, size=1):
        # Sample configurations from the predefined list
        #print("TRACK=>", self.track)
        if size == 1:
            sampled_dict = random.sample(self.predefined_configs, size)
            sample = sampled_dict[0]
            return Configuration(self, values=sample)
        else:
            sample_dicts = random.sample(self.predefined_configs, size)
            samples=[]
            for sample in sample_dicts:
                samples.append(Configuration(self, values=sample))
            return samples
        
class DEHBOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
        self.config_space = None
    def optimize(self):
        
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        def objective(x: Configuration, fidelity: float, **kwargs):
            # Replace this with your actual objective value (y) and cost.
            
            config_dict =self.model_config.cs_to_dict(x)
        
            # start_in= time.time()
            score = self.model_wrapper.run_model(config_dict)
            end = time.time()
            self.logging_util.log(config_dict, (1-score), end-start_time)
            return {"fitness": (1-score), "cost": 1}
        
        n_trials =  self.config['n_trials']
        start_time = time.time()
        output_directory = self.config['output_directory']
        random.seed(self.seed)
        #hyperparameter_dict = self.model_config.get_hyperparam_dict()
        self.config_space = self.create_configspace() 
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        dehb = DEHB(
            f=objective, 
            cs=self.config_space, 
            min_fidelity=1, 
            max_fidelity=10,
            n_workers=1,
            seed=self.seed,
            output_path=output_directory
        )
        self.logging_util.start_logging()
        
        for _ in range(n_trials):
            while True:
                job_info = dehb.ask()
                config = job_info["config"]
                config_dict =self.model_config.cs_to_dict(config)
                if config_dict in self.config_space.predefined_configs:
                    break
            result = objective(job_info["config"], job_info["fidelity"])
            dehb.tell(job_info, result)
        # Run the optimizer
        #traj, runtime, history = dehb.run(fevals = n_trials)
        traj, runtime, history = dehb.traj, dehb.runtime, dehb.history
        best_config = dehb.vector_to_configspace(dehb.inc_config)
        self.best_config = best_config.get_dictionary()
        self.best_value = objective(best_config, 0.0)['fitness']
        total_evaluations = len(history)
        print(f"Evaluated {total_evaluations} configurations")
        print(f"Found best config {self.best_config} with value: {self.best_value}")
        self.logging_util.stop_logging()
        
    def create_configspace(self):
        config_space, param_names, space = self.model_config.get_configspace()
        #print(self.model_config.get_hyperparam_dict())
        combined_space = list(zip(*self.model_config.get_hyperparam_dict().values()))
        config_dict = [dict(zip(param_names, values)) for values in combined_space]
        #random.shuffle(config_dict)
        #to reduce BOHB runtime, we take 20000 samples, this has no impact over accuracy as labels <=50
        #config_dict = config_dict[:20000]
        #print(config_dict)
        cs = CustomConfigurationSpace(config_dict)
        for hyperparameter in config_space.get_hyperparameters():
            cs.add_hyperparameter(hyperparameter)
        # Convert each parameter to a CategoricalHyperparameter
       
        return cs
    