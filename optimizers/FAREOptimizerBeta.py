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

class FAREOptimizerBeta(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed, cluster: Cluster = None) -> None:
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.param_names, self.hyperparameter_space = None, None
        self.cluster = cluster
        self.best_config = None
        self.best_value = None
        self.start = None
        self.step = 0
        self.cluster_access_counts = {}
        self.score_cache = None

    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        
        if not self.cluster:
            raise ValueError("Build cluster first!")
        _, self.param_names, self.hyperparameter_space = self.model_config.get_configspace()
        random.seed(self.seed)
        n_trials =  self.config['n_trials']
        #self.log_performance()
        #import sys
        #sys.exit()
        self.cluster_access_counts = {}
        #initial = int(n_trials/5) if int(n_trials/5)>1 else 2
        initial = 2
        self.logging_util.start_logging()
        self.start = time.time()
        
        best_config= self.find_best_configs(initial,n_trials, self.get_next_budget)
        self.best_config = dict(zip(self.param_names, best_config[:-1]))
        
        self.best_value = self.model_wrapper.test(self.best_config)[-1]
        print(f"Best config is {self.best_config} with performance: {self.best_value}")
        self.logging_util.stop_logging()
        
    def get_next_budget(self, current_budget):
        return current_budget+self.step
    
    def set_cluster(self, cluster):
        self.cluster = cluster
        
    def initialize(self):
        # for binary tree like clustering
        tree_level = 1
        print(f'init lvl:{tree_level}')
        clusters = self.cluster.cache.get(tree_level, None)
        if clusters == None:
            raise ValueError("Initial size need to be altered or sample size needs to be changed")
        return clusters
        
    def initialize_active_learning(self, initial_clusters):
        configs = []
        scores = []
        for initial_cluster, var in initial_clusters:
            cluster_point = initial_cluster.data_sorted_by_mid[0]
            score = self.get_score(cluster_point, self.config['min_budget'])           
            configs.append(cluster_point+[score])
            scores.append(score)
            cluster_id = tuple(cluster_point)
            count, sum, sumsqr  = self.cluster_access_counts.get(cluster_id, (0,0,0))
            self.cluster_access_counts[cluster_id] = (count+1, sum+score, sumsqr+score**2)
        y_column_name = "Score+"
        cols = self.param_names
        cols.append(y_column_name)
        
        data = self.cluster.create_data(cols, configs)
        return data
    

        
    def find_best_leaf(self,acquire, done, best, rest, cluster, time, selection_budget):
        clusters = list(cluster)
        best_cluster = None
        best_UCB_score = None
        best_cluster_index = None
        best_avg = None
        scores_over_cluster = []
        all_points = {}
        indices = random.sample(range(len(clusters[0].data_sorted_by_mid)-2), selection_budget)
        for index, cluster in enumerate(clusters):
            #chunks = [cluster.data_sorted_by_mid[i::selection_budget] for i in range(selection_budget)]
            #points = [random.sample(chunk, 1)[0] for chunk in chunks if chunk]
            points = [cluster.data_sorted_by_mid[i] for i in indices]
            points.append(cluster.data_sorted_by_mid[0])
            best_point_sample = None
            best_point_score = None
            for point in points:
                score = acquire(best.loglike(point, len(done), 2), rest.loglike(point, len(done), 2))
                scores_over_cluster.append(score)
                if not best_point_sample or score>best_point_score:
                    best_point_sample = point
                    best_point_score = score
            all_points[index] = (best_point_sample, best_point_score)
        max_score_over_cluster = max(scores_over_cluster)
        min_score_over_cluster = min(scores_over_cluster)
        #taking data is fine too
        for index, cluster in enumerate(clusters):
            best_point_sample, best_point_score = all_points[index]
            best_point_score = (best_point_score - min_score_over_cluster)/(max_score_over_cluster-min_score_over_cluster)
            
            cluster_variance = cluster.var
            cluster_id = tuple(cluster.data_sorted_by_mid[0])
            
            count, sum, sumsqr = self.cluster_access_counts.get(cluster_id, (0,0,0))
            count+=1
            sum+=best_point_score
            sumsqr+=best_point_score**2
            ucb_score = ucb_with_decay(cluster_variance, sum=sum, sumsq=sumsqr, time = time, selected=count)
            
            if not best_cluster or ucb_score>best_UCB_score:
                best_cluster = cluster
                best_point = best_point_sample
                best_UCB_score = ucb_score
                best_cluster_index = index
                best_avg = best_point_score
            
                    
        best_cluster_id = tuple(best_cluster.data_sorted_by_mid[0])
        count, sum, sumsqr  = self.cluster_access_counts.get(best_cluster_id, (0,0,0))
        self.cluster_access_counts[best_cluster_id] = (count+1, sum+best_avg, sumsqr+best_avg**2)
        #self.cluster_access_counts[best_cluster_id] = (count+1, 0, 0)

                
        return best_cluster, best_point, best_cluster_index
    
    def find_best_configs(self, initial_size, label_budget, budget_alloc_func, rate = 2):
        acquire = self.acquire
        global_best_cluster = None
        global_best_value = None
        initial_clusters = list(self.initialize())
        ##start from here
        data =self.initialize_active_learning(initial_clusters)
        done = data.rows
        # calculating the bracket length
        now_at = int(log2(initial_size) + 1)
        max_height = self.cluster.max_height
        remaining = label_budget-2*initial_size #left for active learning
        tree_remaining = max_height-now_at
        brackets = min(tree_remaining+1, floor((-1 + sqrt(1+8*remaining))/2))
        track_start = now_at-1
        #assuming start from level 1 (2)
        level_step = int(tree_remaining/(brackets-1))
        #level_step = max(now_at,int(max_height - brackets))
        levels_to_traverse = [track_start]
        budgets = [self.config['min_budget']]
        selection_budget = [rate ** brackets]
        self.step = self.calculate_step(brackets)
        
        for bracket in range(brackets-1):
            #level_step+=1
            levels_to_traverse.append(levels_to_traverse[-1]+level_step)
            budgets.append(budget_alloc_func(budgets[-1]))
            selection_budget.append(int(selection_budget[-1]/2))
        #go to furthest level near to leaf        
        current_budget = self.config['min_budget']
        labelled = 0
        results = []
        while True:
            for bracket in range(brackets):
                # lvl_start calculation only works here if starts near leaves
                rounds = brackets-bracket
                best_cluster = None
                best_cluster_index = None
                
                for round in range(rounds): 
                    next_level = levels_to_traverse[bracket+round]
                    current_budget = budgets[bracket+round]
                    #current_budget = None
                    current_selection_budget = selection_budget[bracket+round]
                    clusters = self.cluster.cache.get(next_level, None)
                    if best_cluster:
                        prev_clusters_len = len(clusters_to_explore)
                        current_clusters_len = len(clusters)
                        child_per_cluster = int(current_clusters_len / prev_clusters_len)
                        start_index = child_per_cluster*best_cluster_index
                        end_index = start_index+child_per_cluster
                        clusters = clusters[start_index:end_index]  
                    clusters_to_explore = [node for node, _ in clusters]
                    
                    done = sorted(done, key= lambda x: x[-1], reverse=True)
                    half = floor(floor(sqrt(len(done))))
                    #half = floor(len(done)/2)
                    best= data.clone(done[:half])
                    rest = data.clone(done[half:])
                    labelled+=1
                    best_cluster, best_point, best_cluster_index = self.find_best_leaf(acquire, done, best, rest, clusters_to_explore, labelled, current_selection_budget)
                    
                    score = self.get_score(best_point, current_budget)
            
                    row = best_point+[score]
                    done.append(row)
                    if ((not global_best_cluster or score>global_best_value)):
                        # they may not be globally comparable due to different UCB variances 
                        global_best_cluster = best_cluster
                        #can reduce variance computation
                        global_best_value = score
                    
                        
                    remaining_budget = label_budget - labelled - initial_size 
                    
                   
                        
                    if round==rounds-1:
                        results.append(row) 
                        
                    if remaining_budget == initial_size:
                        break
                
                if remaining_budget == initial_size:
                    break
                      
            if remaining_budget == initial_size:
                break
                
        print(f"Best config found in level {global_best_cluster.lvl} out of {max_height} max levels while starting from {track_start}")
        remaining_budget = label_budget - labelled - initial_size
        for i in range(remaining_budget):
            done = sorted(done, key = lambda x: x[-1], reverse= True)
            indices = random.sample(range(len(global_best_cluster.data.rows)), selection_budget[0]+1)
            points = [global_best_cluster.data.rows[i] for i in indices]
            labelled+=1
            best_configs = self.predict(points, done, 1, data, acquire)
            dones = [config+[self.get_score(config, current_budget)] for config in best_configs]
            done+=dones
            results+=dones
            #done = sorted(done, key = lambda x: x[-1], reverse= True)
        results = sorted(results, key = lambda x: x[-1], reverse= True)
        return results[0]
    
   
    def calculate_step(self, rounds):
        #print("STEP is", )
        #return math.floor((100 - self.model_wrapper.get_data_percentage(self.config["min_budget"]))/(rounds-1))
        return (self.model_wrapper.get_train_size() - self.config["min_budget"])/(rounds-1)
    
    def predict(self, todo, done, budget, data_template, acquire):
        
        half = floor(floor(sqrt(len(done))))
        #half = floor(len(done)/2)
        best= data_template.clone(done[:half])
        rest = data_template.clone(done[half:])
        key = lambda x: acquire(best.loglike(x, len(done), 2), rest.loglike(x, len(done), 2))
        sorted_values = sorted(todo, key = key, reverse=True)
        return sorted_values[:budget]
    
    def acquire(self, best, rest):
        #bs = math.exp(best)
        #rs = math.exp(rest)
        #print("bs:",best)
        #score = bs/(bs+rs)
        score = best 
        return score
    
    def get_score(self, leaf_mid, budget=None):
        hyperparams = dict(zip(self.param_names, leaf_mid))
        score = self.model_wrapper.run_model(hyperparams, budget)
        self.logging_util.log(hyperparams, 1-score, (time.time() - self.start))
        return score
