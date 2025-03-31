import ast
from functools import reduce
import itertools
import random
import numpy as np
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace import ConfigurationSpace
from ConfigSpace import EqualsCondition, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause
class ModelConfiguration():
    def __init__(self, config, conditions = None, constraints = None, seed = 42, needed = 200000):
        self.config = config
        self.seed = seed
        # for API specific representation of hyperparameters
        self.configspace = None
        # for simpler representation of hyperparameters
        self.param_names = None
        self.hyperparam_space = None
        self.hyperparam_dict = None
        self.conditions = conditions
        self.constraints = constraints
        self.get_configspace(needed)
        
    def set_seed(self, seed):
        self.seed = seed
    
    def get_hyperparam_dict(self):
        return self.hyperparam_dict
    
    def get_hyperconfig_distribution(self):
        hyperparameters = {}
        cs = ConfigurationSpace()
        cs.seed(self.seed)
        forbiddens = []
        constraints = []
        for param_name, param_range in self.config.items():
            if isinstance(param_range, list):
                if all(isinstance(x, str) for x in param_range):
                    # Return the list of strings unchanged if all elements are strings
                    hp = CategoricalHyperparameter(param_name, param_range)
                
                elif len(param_range) >= 2:
                    start, end = param_range
                    
                    # Check if both start and end are integers
                    if isinstance(start, int) and isinstance(end, int):
                        hp = UniformIntegerHyperparameter(param_name, start, end)
                    
                    # Check if both start and end are floats
                    elif isinstance(start, float) and isinstance(end, float):
                        hp = UniformFloatHyperparameter(param_name, start, end)
                    
                    # If they are mixed types or not valid, raise an error
                    else:
                        raise ValueError("Both start and end must be of the same type (either int or float).")
            
                else:
                    raise ValueError("param_range must be a list of two numeric elements or multiple string elements.")
        
            else:
                raise ValueError("param_range must be a list.")
            
            hyperparameters[param_name] = hp
        
            # Add conditions dynamically from YAML
        if self.conditions:
            for condition in self.conditions:
                param = condition["parameter"]
                rules = condition["rules"]
                
                for rule in rules:
                    if_expr = rule["if"]
                    parent_param, _, parent_value = if_expr.split()
                    parent_value = parent_value.strip("'")
                    
                    # Add conditions based on rule type
                    if "valid_values" in rule:
                        valid_values = rule["valid_values"]
                        remaining_options = [val for val in hyperparameters[param].choices if val not in valid_values]
                        if remaining_options:
                            restricted = ForbiddenAndConjunction(
                                ForbiddenEqualsClause(hyperparameters[parent_param], parent_value),
                                ForbiddenInClause(hyperparameters[param], remaining_options),
                            )
                            hyperparameters[param].default_value = valid_values[0]
                            hyperparameters[parent_param].default_value = parent_value
                            forbiddens.append(restricted)

        if self.constraints:
            for constraint in self.constraints:
                param = constraint["parameter"]
                rules = constraint["rules"]
                
                for rule in rules:
                    if_expr = rule["if"]
                    parent_param, _, parent_value = if_expr.split()
                    parent_value = parent_value.strip("'")
                    
                    # Add conditions based on rule type
                    if "active" in rule:
                        actives = rule["active"]
                        for active in actives:
                            cons = EqualsCondition(hyperparameters[active], hyperparameters[parent_param], parent_value) 
                            constraints.append(cons)
        for param_name, hyperparameter_config in hyperparameters.items():
            cs.add(hyperparameter_config)
        if forbiddens:
            cs.add(forbiddens)
        if constraints:
            cs.add(constraints)
        print(cs)
        return cs
    
    def __expand_range(self, param_range):
        """Expand a range defined by a list of two numbers into a list of integers or floats."""
        if isinstance(param_range, list):
            if all(isinstance(x, str) for x in param_range):
                # Return the list of strings unchanged if all elements are strings
                return param_range
            
            elif len(param_range) == 3:
                start, end, step = param_range
                
                # Check if both start and end are integers
                if isinstance(start, int) and isinstance(end, int):
                    # Generate a list of 20 evenly spaced integers
                    return [i for i in range(start, (end+1), step)]
                
                # Check if both start and end are floats
                elif isinstance(start, float) and isinstance(end, float):
                    # Generate a list of 50 floats
                    return np.round(np.linspace(start, end, step), 4).tolist()
                
                # If they are mixed types or not valid, raise an error
                else:
                    raise ValueError("Both start and end must be of the same type (either int or float).")
        
            else:
                raise ValueError("param_range must be a list of two numeric elements or multiple string elements.")
    
        else:
            raise ValueError("param_range must be a list.")
        
    def __create_all_configspace(self, expanded_config):
        # Generate all combinations of hyperparameters
        hyperparameter_space = list(itertools.product(*expanded_config.values()))
        return expanded_config, list(expanded_config.keys()), hyperparameter_space
    
    def get_configspace(self, needed = 50000, recompute = False):
        if recompute or not all([self.configspace, self.param_names, self.hyperparam_space]):
            cs = self.get_hyperconfig_distribution()
            if cs:
                sample = set()
                while len(sample) < needed:
                    # get random needed params
                    config = cs.sample_configuration().get_dictionary()
                    sample.add(frozenset(config.items()))
                
                unique_config_list = [dict(config) for config in sample]
                
                # Extract parameter names and values
                self.param_names = list(unique_config_list[0].keys())  # All configurations share the same keys
                self.hyperparam_space = [[config[param] for param in self.param_names] for config in unique_config_list]
                self.configspace = cs
                param_values = {param: [] for param in  self.param_names}
                for param_set in self.hyperparam_space:
                    for i, value in enumerate(param_set):
                        param_values[self.param_names[i]].append(value)
                self.hyperparam_dict = param_values
            
        return self.configspace, self.param_names, self.hyperparam_space
    
    def cs_to_dict(self, config):
            config_str = str(config)
            start = config_str.index('{')
            end = config_str.rindex('}')
            return ast.literal_eval(config_str[start:end+1])               
                    
    def get_configspace_deprecated(self, needed = 50000, recompute = False):
        if recompute or not all([self.expanded_configs, self.param_names, self.hyperparam_space]):
            sample = set()
            expanded_config = {key: self.__expand_range(value) for key, value in self.config.items()}
            combos = reduce(lambda x, y: x * y, (len(v) for v in expanded_config.values()), 1)
            # if all combos are smaller, return it
            if combos<needed:
                return self.__create_all_configspace(expanded_config)
            
            while len(sample) < needed:
                # get random needed params
                elem = tuple([random.choice(value) for key, value in expanded_config.items()])
                sample.add(elem)
                
            hyperparameter_space = list(sample)
            self.expanded_configs = expanded_config
            self.param_names = list(expanded_config.keys())
            self.hyperparam_space = hyperparameter_space
            
        return self.expanded_configs, self.param_names, self.hyperparam_space