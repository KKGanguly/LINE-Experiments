from math import ceil, sqrt
import random
from typing import Dict, Generator, List
from ezr_24Aug14.ezr import CLUSTER, DATA

class Cluster:
    def __init__(self, param_names, hyperparameter_space):
        self.clusters: Dict[str, Generator[CLUSTER, None, None]] = {}
        self.param_names = param_names
        self.hyperparameter_space = hyperparameter_space
        self.data = self.create_data(self.param_names, self.hyperparameter_space)
        
    def get_clustered_data(self, stop = -1, expand = None):
        def search_node(node):
            return (node, self.clusters.get(str(node.data.rows), None)) if node else (None, None)
        
        data = expand.data if expand else self.data
        node, value= search_node(expand)
        #print("found:",(str(node.data.rows) if node else ""))
        if not value:
            node = self.__cluster(data, stop)
            value = list(self._get_leaf(node))
            self.clusters[str(node.data.rows)] = value
       
        return value
        
    def _get_leaf(self, cluster):
        for node,isLeaf in cluster.nodes():
            print(node)
            print("lvl", node.lvl)
            if isLeaf: 
                yield node
        
                       
    def leafp(self, x): return x.lefts==None or x.rights==None
    
    def __cluster(self, data, stop  = -1):
        stop = ceil(len(data.rows)/stop) if stop>0 else ceil(sqrt(len(data.rows)))
        cluster = data.cluster(data.rows, sortp = False, stop = stop)
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
        random.seed(seed)
        random_rows = random.sample(node.data.rows, min(len(node.data.rows), 5))
        variance = sum(node.data.dist(node.mid, row) ** 2 for row in random_rows) / (5 - 1)
        return variance
        