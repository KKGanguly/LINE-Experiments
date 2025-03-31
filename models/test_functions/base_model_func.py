from models.base_model import BaseModel
import numpy as np

class BaseModelFunc(BaseModel):
    def __init__(self,config):
        super().__init__(config, 0)
        
    def evaluate(self, hyperparams):
        def ackley(a=20, b=0.2, c=2*np.pi):
            x = hyperparams.get('a', 0)
            y = hyperparams.get('b', 0)
            term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
            term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
            return (term1 + term2 + a + np.exp(1))
        def normalize_ackley(f_max):
            """Normalize the Ackley function output to range [0, 1]."""
            f_value = ackley()
            f_min = 0  # Known minimum of the Ackley function
            f_normalized = (f_value - f_min) / (f_max - f_min)
            return 1-f_normalized
        return normalize_ackley(50)
    