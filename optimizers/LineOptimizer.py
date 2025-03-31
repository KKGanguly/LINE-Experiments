import hashlib
from itertools import chain
from math import ceil, floor, log2, sqrt
import math
import os
import random
from time import sleep
import time
from models.test_functions.base_model_func import BaseModelFunc
from models.test_functions.optimization_test_func_wrapper import OptimizationTestFuncWrapper
import numpy as np
from optimizers.base_optimizer import BaseOptimizer
from utils.UCB import ucb, ucb_with_decay
from utils.clustering_tree_faster import Cluster
import pandas as pd

class LineOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed) -> None:
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.param_names, self.hyperparameter_space = None, None
        self.best_config = None
        self.best_value = None
        self.start = None 
    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        
        self.start = time.time()
        self.logging_util.start_logging()
        _, self.param_names, self.hyperparameter_space = self.model_config.get_configspace()
        random.seed(self.seed)
        n_trials =  self.config['n_trials']
        indices = random.sample(range(len(self.hyperparameter_space)), 50000)
        hyperparams = [self.hyperparameter_space[i] for i in indices]
        min_vals = [float('inf')] * len(hyperparams[0])
        max_vals = [float('-inf')] * len(hyperparams[0])

        
        for param_set in hyperparams:
            for i, value in enumerate(param_set):
                if isinstance(value, (int, float)):
                    if value < min_vals[i]:
                        min_vals[i] = value
                    if value > max_vals[i]:
                        max_vals[i] = value
                        
        data = Data(rows=hyperparams, samples=self.config['samples'], min_vals=min_vals, max_vals=max_vals)
        selected = data.around(n_trials)            
        scores = {}
        for mid in selected:
            score = self.get_score(mid)
            scores[tuple(mid)] = score
            
        self.best_config, self.best_value = max(scores.items(), key=lambda item: item[1])
        
        print(f"Best config is {self.best_config} with performance: {self.best_value}")
        self.logging_util.stop_logging()
        
    def get_score(self, mid):
        hyperparams = dict(zip(self.param_names, mid))
        score = self.model_wrapper.run_model(hyperparams)
        self.logging_util.log(hyperparams, 1-score, (time.time() - self.start))
        return score  

class Data:
    def __init__(self, rows, samples, min_vals, max_vals):
        self.rows = rows  # List of data points
        self.samples = samples  # Number of samples to consider
        self.min_vals = min_vals  # Min values for normalization
        self.max_vals = max_vals
    
    
    def any(self, rows, num=1):
        """Randomly pick an item from the rows, assuming a shuffled list"""
        indices = random.sample(range(len(rows)), num)
        selected = [rows[i] for i in indices]
        return selected if num > 1 else selected[0]

    def min_item(self, lst, key_func):
        """Find the item that minimizes the key function."""
        return min(lst, key=key_func)

    def pick(self, u):
        """Stochastically pick an item based on weighted probabilities."""
        total = sum(u.values())
        r = random.uniform(0, total)
        for key, value in sorted(u.items(), key=lambda item: item[1], reverse=True):
            r -= value / total
            if r <= 0:
                return key
        return random.choice(list(u.keys()))  # Fallback if no choice made

    def around(self, k, rows=None):
        """
        Select k centroids based on LINE algorithm (similar to k-means++).
        
        :param k: Number of centroids to select
        :param rows: List of data points (if None, use self.rows)
        :return: List of k centroids
        """
        if rows is None:
            rows = self.rows

        out = [self.any(rows)]  # Select the first centroid randomly
        for index in range(2, k + 1):
            u = {}
            remaining_rows = [row for row in rows if row not in out]
            random_candidates = self.any(remaining_rows, min(self.samples, len(remaining_rows)))
            for i in range(min(self.samples, len(remaining_rows))):
                r1 = random_candidates[i]
                r2 = self.min_item(out, lambda ru: self.xdist(r1, ru))  # Find closest centroid
                if r1 and r2:
                    u[tuple(r1)] = self.xdist(r1, r2) ** 2  # Store squared distance
            out.append(self.pick(u))  # Stochastically pick the next centroid
            print(f'{index} out of {k} centroids selected')
        return out
    
    def normalize(self, value, feature_index):
            """Normalize a numerical value between 0 and 1."""
            if value == "?":
                return "?"

            min_val, max_val = self.min_vals[feature_index], self.max_vals[feature_index]
            return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

    def dist(self, a, b, index):
        if a == "?" and b == "?":
            return 1 
        if isinstance(a, str) and isinstance(b, str):
            return 0 if a == b else 1
        a, b = self.normalize(a, index), self.normalize(b, index)
        if a == "?":
            a = 1 if b < 0.5 else 0  # Handle missing a
        if b == "?":
            b = 1 if a < 0.5 else 0  # Handle missing b
        return abs(a - b)

    def xdist(self, p1, p2, p =2):
        #return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
        n = 0
        d = 0
        for a, b in zip(p1, p2):
            d += abs(self.dist(a, b, n)) ** p
            n += 1
        return (d / n) ** (1 / p) if n > 0 else 0
            