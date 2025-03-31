from collections import deque
from math import ceil, sqrt
import random
from typing import Dict, Generator, List
from ezr_24Aug14.ezr import CLUSTER, DATA

class Cluster:
    def __init__(self, model_config, seed = 1):
        self.clusters: Dict[str, Generator[CLUSTER, None, None]] = {}
        _, self.param_names, self.hyperparameter_space = model_config.get_configspace()
        self.seed = seed
        self.data = self.create_data(self.param_names, self.hyperparameter_space)
        self.cluster = self.__cluster(self.data)
        self.max_height = self.get_max_height()
        self.cache = {}
        self.build_cache()
        
        
    def set_seed(self, seed):
        self.seed = seed
        
    def get_max_height(self):
        node = max(self.get_all_leaf(), key = lambda node: node.lvl)
        return node.lvl
    
    def get_all_leaf(self):
        yield from self.get_leaf_by_node(self.cluster)
    
    def get_leaf_by_node(self, node):
        for node,isLeaf in node.nodes():
            if isLeaf: 
                yield node  
    
    def build_cache(self):
        for node,isLeaf in self.cluster.nodes():
            self.cache[node.lvl] = self.cache.get(node.lvl, []) + [(node,self.get_cluster_variance(node))]  
 
    def get_nodes_by_lvl(self, node, lvl):
        def leafp(x): return x.lefts is None and x.rights is None

        queue = deque([(node, leafp(node))])
        
        while queue:
            node, isLeaf = queue.popleft()

            if node.lvl > lvl: break
        
            if node.lvl == lvl: 
                yield node
            
            if node.lefts:
                queue.append((node.lefts, leafp(node.lefts)))
            if node.rights:
                queue.append((node.rights, leafp(node.rights)))
        return None   
    
    def leafp(self, x): return x.lefts==None or x.rights==None
    
    def __cluster(self, data, stop  = -1):
        stop = ceil(len(data.rows)/stop) if stop>0 else ceil(sqrt(len(data.rows)))
        cluster = data.cluster(data.rows, sortp = False, stop = stop, seed = self.seed)
        return cluster
    
    
    def create_data(self, cols, rows):
        def format_param_names():
            formatted_param_names = []
            for i, value in enumerate(rows[0]):
                # If the value is a string, convert the corresponding param_names entry to lowercase
                if isinstance(value, str):
                    formatted_param_names.append(cols[i].lower())
                # Otherwise, convert it to start with an uppercase letter
                else:
                    formatted_param_names.append(cols[i].capitalize())
            return formatted_param_names
        def row_gen():
            yield format_param_names()
            for row in rows:
                yield list(row)
        return DATA().adds(row_gen())
        
    def get_data(self):
        return self.data
    
    def get_cluster_variance(self, node, seed=10):
        """Get cluster variance (maximum distance within a cluster) using euclidean distance functions."""
        random.seed(self.seed)
        random_rows = random.sample(node.data.rows, min(len(node.data.rows), 5))
        variance = sum(node.data.dist(node.mid, row) ** 2 for row in random_rows) / (5 - 1)
        return variance
        