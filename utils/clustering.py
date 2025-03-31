# Cluster hyperparameter space
import random

# DISTANCE: between two symbols
def dist_sym(a, b):
    if a == b == "?":
        return 1
    return 1 if a != b else 0

# DISTANCE: between two numbers
def dist_num(a, b, min_val=0, max_val=1):
    if a == b == "?":
        return 1
    a = (a - min_val) / (max_val - min_val) if a != "?" else (1 if b < 0.5 else 0)
    b = (b - min_val) / (max_val - min_val) if b != "?" else (1 if a < 0.5 else 0)
    return abs(a - b)

# DISTANCE: between two rows (Minkowski)
def xDist(row1, row2, cols, p=2):
    d = 0
    for col in cols:
        a = row1[col]
        b = row2[col]
        # Assuming columns are either numeric or symbolic
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            d += dist_num(a, b) ** p
        else:
            d += dist_sym(a, b) ** p
    return (d ** (1 / p)) / len(cols) ** (1 / p)

# Helper function: Calculate the mean (centroid) of a cluster of rows (numerical columns only)
def calculate_centroid(cluster, cols):
    n = len(cluster)
    centroid = [sum(row[i] for row in cluster) / n for i in cols]
    return centroid

# KMeans function: Perform KMeans clustering on a list of rows (matrix)
def kmeans(matrix, k=10, n=10, cols=None):
    def loop(n, centroids):
        clusters = {i: [] for i in range(k)}  # Dictionary to store clusters
        for row in matrix:
            # Find the closest centroid for each row
            closest_centroid_index = min(range(len(centroids)), key=lambda i: xDist(centroids[i], row, cols))
            clusters[closest_centroid_index].append(row)
        # If it's the last iteration, return the clusters
        if n == 0:
            return list(clusters.values())
        
        # Otherwise, recalculate centroids and continue
        new_centroids = [calculate_centroid(cluster, cols) for cluster in clusters.values() if cluster]
        
        return loop(n-1, new_centroids)
    
    # Randomly shuffle the rows and select initial centroids
    random.shuffle(matrix)
    initial_centroids = matrix[:k]
    
    # Run the KMeans loop
    return loop(n, initial_centroids)



def cluster_Xs(x_space, n_clusters):
    """Clustering using X values with custom KMeans."""
    # Run KMeans clustering
    #print(x_space)
    clusters = kmeans(x_space, k=n_clusters, n=10, cols=list(range(len(x_space[0]))))
    # Get cluster centroids and assign clusters
    centroids = [calculate_centroid(cluster, range(x_space.shape[1])) for cluster in clusters]
    labels = [0] * len(x_space)
    
    for cluster_index, cluster in enumerate(clusters):
        for row in cluster:
            labels[x_space.tolist().index(row)] = cluster_index
            
    return centroids, labels

def calculate_cluster_variances(X_space, labels, n_clusters):
    """Get cluster variance (maximum distance within a cluster) using custom distance functions."""
    variances = []
    
    for cluster in range(n_clusters):
        cluster_points = [X_space[i] for i in range(len(labels)) if labels[i] == cluster]
        
        max_distance = 0
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                distance = xDist(cluster_points[i], cluster_points[j], range(X_space.shape[1]))
                max_distance = max(max_distance, distance)
        
        variances.append(max_distance)
    
    return variances
