import numpy as np

def k_means(X, k, max_iters=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Example usage:
# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 2)

# Define number of clusters
k = 3

# Run K-means algorithm
centroids, labels = k_means(X, k)

print("Final centroids:")
print(centroids)
print("Labels:")
print(labels)
