import unittest
from ezr_24Aug14.ezr import DATA  # Import the necessary classes
from utilities.clustering_tree import Cluster

class TestCluster(unittest.TestCase):

    def setUp(self):
        self.param_names = ["Param1", "Param2"]
        self.hyperparameter_space = [(0,0),(1, 1), (1, 2), (3, 4), (5, 6)]
        self.stop = 2
        
        # Create actual instances of DATA and Cluster
        self.cluster = Cluster(self.param_names, self.hyperparameter_space, self.stop)
        print("here")
        #print([l.left for l in self.cluster.get_clustered_data()])
        
    def test_initialization(self):
        """Test that data is correctly initialized in the cluster."""
        self.assertIsInstance(self.cluster.data, DATA)
        formatted_names = [name.lower() if isinstance(value, str) else name.capitalize() 
                           for name, value in zip(self.param_names, self.hyperparameter_space[0])]
        self.assertEqual(self.cluster.data.cols.names, formatted_names)

    def test_get_clustered_data_initializes_cluster(self):
        """Test that get_clustered_data initializes clusters and yields leaves."""
        clustered_data = list(self.cluster.get_clustered_data())

        # Ensure that clustered data is not empty
        self.assertTrue(clustered_data)
        # Assuming the yielded items are leaves and have the expected structure
        for leaf in clustered_data:
            self.assertTrue(self.cluster.leafp(leaf))

    def test_get_clustered_data_with_expand(self):
        """Test that get_clustered_data updates cluster on expand."""
        # Here you need to define a valid expand key based on your clustering logic.
        # Assuming that the first hyperparameter space row corresponds to the key.
        expand_key = (1, 2)
        result = list(self.cluster.get_clustered_data(expand=expand_key))
        
        # Ensure that expanding the cluster retrieves leaves
        self.assertTrue(len(result) > 0)

    def test_get_cluster_variance(self):
        """Test get_cluster_variance computes distances correctly."""
        # Assuming the first leaf node can be used for this test
        node = next(self.cluster.get_clustered_data())
        variance = self.cluster.get_cluster_variance(node)
        
        # Assuming you have a way to calculate the expected variance based on node's data
        expected_variance = self.calculate_expected_variance(node)  # Placeholder for expected variance calculation
        self.assertEqual(variance, expected_variance)

    def calculate_expected_variance(self, node):
        """Placeholder for a method to calculate expected variance."""
        # You need to implement the logic to calculate the expected variance based on the node's data
        # For now, let's assume a dummy return value
        return 1  # Example expected variance value for the sake of testing

if __name__ == "__main__":
    unittest.main()
