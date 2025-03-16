import numpy as np 

class Kmeansclustering:
    def __init__(self, X, K=10, max_iters=100, tol=1e-5):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.X = X
        self.n_samples, self.n_features = X.shape

    def predict(self, X):
        cluster_labels = np.zeros(self.n_samples)
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        k_list = np.random.choice(self.n_samples, self.K, replace=False)
        for k in k_list:
            self.centroids.append(self.X[k])

        for _ in range(self.max_iters):
            self.clusters = self.clustering(self.centroids)
            old_centroids = self.centroids
            self.centroids = self.new_centroids(self.clusters)
            distances = []
            for cent_pt in range(self.K):
                diff = np.sqrt(np.sum((self.centroids[cent_pt] - old_centroids[cent_pt]) ** 2))
                distances.append(diff)
            if sum(distances) < self.tol:
                break

        for clust_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                cluster_labels[sample_idx] = clust_idx

        return cluster_labels

    def clustering(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for sample_idx, sample in enumerate(self.X):
            distances = []
            for cent in centroids:
                dist = np.sqrt(np.sum((sample - cent) ** 2))
                distances.append(dist)
            min_dist = np.argmin(distances)
            clusters[min_dist].append(sample_idx)
        return clusters

    def new_centroids(self, clusters):
        centroids = []
        k_list = np.random.choice(self.n_samples, self.K, replace=False)
        for k in k_list:
            centroids.append(self.X[k])

        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids
