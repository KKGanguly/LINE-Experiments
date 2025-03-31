import random
from models.model_wrapper import ModelWrapper


class BaseOptimizer:
    def __init__(self, config, model_wrapper: ModelWrapper, model_config, logging_util = None, seed =42) -> None:
        self.config = config
        self.model_wrapper = model_wrapper
        self.model_config = model_config
        self.seed = seed
        self.logging_util = logging_util
        
    def optimize(self):
        raise NotImplementedError("This method must be implemented by subclass!!")
    
    def set_model_config(self, model_config):
        self.model_config = model_config
    
    def set_logging_util(self, logging_util):
        self.logging_util = logging_util
        
    def set_seed(self, seed):
        self.seed =seed
        random.seed(seed)
    def evaluate(self):
        pass