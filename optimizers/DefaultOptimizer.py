from optimizers.base_optimizer import BaseOptimizer


class DefaultOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed) -> None:
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = {}
        self.best_value = None
    def optimize(self):
        self.best_value = 0