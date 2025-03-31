from math import ceil, sqrt
from ezr_24Aug14.ezr import DATA

class Cluster:
    def __init__(self, param_names, hyperparameter_space, stop):
        self.cluster = None
        self.stop = stop
        self.param_names = param_names
        self.hyperparameter_space = hyperparameter_space
        self.data = self.__create_data()
        
    def get_clustered_data(self, expand = None):
        def search_node():
            for node,isLeaf in self.cluster.nodes():
                if repr(node) == repr(expand):
                    return node
            return None
        def update_node_in_place(node, new_node):
            """Copy attributes from new_node to node in place."""
            for attr in ['data', 'left', 'right', 'lefts', 'rights', 'cut', 'mid', 'fun', 'lvl']:
                setattr(node, attr, getattr(new_node, attr))
        if not self.cluster:
            self.cluster = self.__cluster(self.data)
        result = self.cluster
        if expand and (node := search_node()):
            if self.leafp(node):
                new_node = self.__cluster(node.data)
                update_node_in_place(node, new_node)
            result = node
        yield from self._get_leaf(result)
                
    def _get_leaf(self, cluster):
        for node,isLeaf in cluster.nodes():
            if isLeaf: 
                yield node
                       
    def leafp(self, x): return x.lefts==None or x.rights==None
    
    def __cluster(self, data):
        cluster = data.cluster(data.rows, sortp = False, stop = len(data.rows)/self.stop)
        return cluster
    
    
    def __create_data(self):
        def format_param_names():
            formatted_param_names = []
            for i, value in enumerate(self.hyperparameter_space[0]):
                # If the value is a string, convert the corresponding param_names entry to lowercase
                if isinstance(value, str):
                    formatted_param_names.append(self.param_names[i].lower())
                # Otherwise, convert it to start with an uppercase letter
                else:
                    formatted_param_names.append(self.param_names[i].capitalize())
            return formatted_param_names
        def row_gen():
            yield format_param_names()
            for row in self.hyperparameter_space:
                yield list(row)
        return DATA().adds(row_gen())
        

    def get_cluster_variance(self, node):
        """Get cluster variance (maximum distance within a cluster) using euclidean distance functions."""
        if self.leafp(node):
            one, two = node.data.twoFar(node.data.rows, sortp=False)
            return node.data.dist(one, two)
        return node.dist(node.left, node.right)