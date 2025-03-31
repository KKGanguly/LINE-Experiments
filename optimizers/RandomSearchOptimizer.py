import random
import time
from optimizers.base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed) -> None:
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.param_names, self.hyperparameter_space = None, None
        self.best_config = None
        self.best_value = None
    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        self.best_value = 0
        _, self.param_names, self.hyperparameter_space = self.model_config.get_configspace()
        param_combinations = [dict(zip(self.param_names, combination)) for combination in self.hyperparameter_space]
        random.seed(self.seed)
        self.logging_util.start_logging()
        self.start = time.time()
        random.shuffle(param_combinations)
        for i in range(self.config['n_trials']):
            selected = param_combinations[i]
            print(selected)
            score = self.model_wrapper.run_model(selected)
            self.logging_util.log(selected, 1-score, (time.time() - self.start))
        self.best_config = selected
        self.best_value = 1-score
        print(f"Best config is {self.best_config} with performance: {self.best_value}")
        self.logging_util.stop_logging()