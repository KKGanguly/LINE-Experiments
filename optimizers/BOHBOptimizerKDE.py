import ast
import os
import random
import shutil
import time
from optimizers.base_optimizer import BaseOptimizer
from smac import Scenario

from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from smac import MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband
import argparse
import numpy as np
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from ConfigSpace.configuration import Configuration

class CustomConfigurationSpace(ConfigurationSpace):
    def __init__(self, predefined_configs, mapping = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predefined_configs = predefined_configs
        self.mapping = mapping
    
    def set_mapping(self, mapping):
        self.mapping = mapping
        
    def sample_configuration(self, size=1):
        # Sample configurations from the predefined list
        #print("TRACK=>", self.track)
        if size == 1:
            sampled_dict = random.sample(self.predefined_configs, size)
            sample = sampled_dict[0]
            encoded = {}
            for key, val in sample.items():
                if isinstance(val, str):
                    encoded[key] = self.mapping[key].index(val)  
                else: encoded[key] = val
            return Configuration(self, values=encoded)
        else:
            sample_dicts = random.sample(self.predefined_configs, size)
            samples=[]
            for sample in sample_dicts:
                print(sample)
                samples.append(Configuration(self, values=sample))
            return samples

class MyWorker(Worker):

    def __init__(self,  *args, eval, label_budget, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval = eval
        self.label_budget = label_budget
    def compute(self, config, budget, **kwargs):
        #config_dict =self.model_config.cs_to_dict(config)
        self.label_budget -= 1
        score = self.eval(config, budget, self.label_budget)
        return({
                    'loss': float(1 - score),  # this is the a mandatory field to run hyperband
                    'info': score  # can be used for any user-defined information - also mandatory
                })
        
    
class BOHBOptimizerKDE(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = -1
        self.measured_time = None
    def optimize(self):
        def encode_config_space(config_space):
            #config_space, param_names, space = self.model_config.get_configspace()
            #config_dict = [dict(zip(param_names, values)) for values in space]
            encoded_space = ConfigurationSpace()
            #mapping = {}
            for hyperparameter in config_space.get_hyperparameters():
                if isinstance(hyperparameter, CategoricalHyperparameter):
                    #mapping[hyperparameter.name] = list(hyperparameter.choices)
                    choices = list(range(len(hyperparameter.choices)))
                    encoded_hyper = UniformIntegerHyperparameter(hyperparameter.name, lower=min(choices), upper=max(choices))
                else:
                    encoded_hyper = hyperparameter
                encoded_space.add_hyperparameter(encoded_hyper)
            #print(encoded_space)
            #encoded_space.set_mapping(mapping=mapping)
            
            return encoded_space


        def decode_config(encoded_config, original_config_space):
            """
            Decodes an encoded configuration back to its original categorical values.
            """
            decoded_config = encoded_config.copy()
            for hyperparameter in original_config_space.get_hyperparameters():
                if isinstance(hyperparameter, CategoricalHyperparameter):
                    choices = hyperparameter.choices
                    decoded_config[hyperparameter.name] = choices[encoded_config[hyperparameter.name]]
                    if "model_penalty" in decoded_config and "model_solver" in decoded_config:
                        model_penalty = decoded_config["model_penalty"]
                        if model_penalty == 'l1' and decoded_config["model_solver"] not in ['liblinear', 'saga']:
                            # If model_penalty is 'l1', and model_solver is invalid, suggest again
                            decoded_config["model_solver"] = random.choice(['liblinear', 'saga'])
                        elif model_penalty == 'l2' and decoded_config["model_solver"] not in ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
                            # If model_penalty is 'l2', and model_solver is invalid, suggest again
                            decoded_config["model_solver"] = random.choice(['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
            return decoded_config
        
        def objective(config, budget, label_budget):
              
            decoded_config = decode_config(config, cs)
            #print("decoded:")
            #print(decoded_config)
            score = self.model_wrapper.run_model(decoded_config, budget)
            #logging does not work for last few iters
            if score>self.best_value:
                self.best_config = decoded_config
                self.best_value=score
            if label_budget>=0:
                elp_time =  (time.time()-start_time)
                self.measured_time=elp_time
                self.logging_util.log(decoded_config, (1-score),elp_time)
        
            return score
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
       
        start_time = time.time()
        output_directory = self.config['output_directory']
        #cs, _, _ = self.model_config.get_configspace()
        #hyperparameter_dict = self.model_config.get_hyperparam_dict()
        max_budget = int(self.model_wrapper.get_train_size())
        cs = self.create_configspace()
        encoded_cs = encode_config_space(cs)
        parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
        parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=self.config['min_budget'])
        parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=max_budget)
        #parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=self.config['n_trials'])
        args=parser.parse_args()
        NS = hpns.NameServer(run_id='bohb', host='127.0.0.1', port=None)
    
        NS.start()
        w = MyWorker(eval = objective, label_budget= self.config['n_trials'], nameserver='127.0.0.1',run_id='bohb')
        w.run(background=True)
        bohb = BOHB( configspace = encoded_cs,
            run_id = 'bohb', nameserver='127.0.0.1',
            min_budget=args.min_budget, max_budget=args.max_budget, 
            result_logger=None,
        )
        self.logging_util.start_logging()
        res = bohb.run(n_iterations=1) #we do not use n_iter so any large value is ok
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

        # Run the optimizer
        #incumbent = res.get_incumbent_id()
        #inc_runs = res.get_runs_by_id(incumbent)
        #inc_run = inc_runs[-1]
        #id2config = res.get_id2config_mapping()
        #total_evaluations = len(id2config.keys())
        #self.best_config = decode_config(id2config[incumbent]['config'], cs)
        #self.best_value = inc_run.loss
        print(f"Evaluated {self.config['n_trials']} configurations")
        print(f"Found best config {self.best_config} with value: {1-self.best_value}")
        self.logging_util.stop_logging()
    
    
    def create_configspace(self):
        config_space, param_names, space = self.model_config.get_configspace()
        config_dict = [dict(zip(param_names, values)) for values in space]
        #random.shuffle(config_dict)
        #to reduce BOHB runtime, we take 20000 samples, this has no impact over accuracy as labels <=50
        #config_dict = config_dict[:20000]
        cs = CustomConfigurationSpace(config_dict)
        for hyperparameter in config_space.get_hyperparameters():
            cs.add_hyperparameter(hyperparameter)
        # Convert each parameter to a CategoricalHyperparameter
       
        return cs
