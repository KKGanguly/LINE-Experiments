import ast
from functools import reduce
import itertools
import random
import numpy as np
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace import ConfigurationSpace
from ConfigSpace import EqualsCondition, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause

class ModelConfigurationStatic():
    def __init__(self, config, seed = 42):
        self.config = config
        self.seed = seed
        # for API specific representation of hyperparameters
        self.configspace = None
        # for simpler representation of hyperparameters
        self.param_names = None
        self.hyperparam_space = None
        self.hyperparam_dict = None
        self.get_configspace()
        
    def set_seed(self, seed):
        self.seed = seed
    
    def get_hyperparam_dict(self):
        return self.hyperparam_dict
    
    def get_hyperconfig_distribution(self):
        cs = ConfigurationSpace()
        """
        with open("config_items.txt", "w") as file:
            for param_name, param_values in self.config.items():
                file.write(f"{param_name}: {param_values}\n")
        """     
        for param_name, param_values in self.config.items():
            if isinstance(param_values, (list, set)):
                param_values = list(set(param_values))  # Ensure it's a list
                hp = CategoricalHyperparameter(param_name, param_values)
                cs.add_hyperparameter(hp)
            else:
                raise ValueError(f"Values for parameter '{param_name}' must be a list or set.")
        cs.seed(self.seed)
        return cs
    
    
    def get_configspace(self, recompute = False):
        if recompute or not all([self.configspace, self.param_names, self.hyperparam_space]):
            self.configspace = self.get_hyperconfig_distribution()
            self.param_names = list(self.config.keys())
            self.hyperparam_space = [[value for value in config_values] for config_values in self.config.values()]
            param_values = {param: [] for param in  self.param_names}
            for index, param_set in enumerate(self.hyperparam_space):
                param_values[self.param_names[index]].extend(param_set)
            self.hyperparam_dict = param_values
        return self.configspace, self.param_names, self.hyperparam_space
    
    def cs_to_dict(self, config):
        config_str = str(config)
        start = config_str.index('{')
        end = config_str.rindex('}')
        return ast.literal_eval(config_str[start:end+1])               
                    