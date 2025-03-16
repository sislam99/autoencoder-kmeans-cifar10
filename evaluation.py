import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from validclust import dunn

def evaluate_silhouette(latent_TEST_DATA, cluster_range):
    for k in cluster_range:
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_xlim([-0.3, 1.0])
        ax.set_ylim([0, len(latent_TEST_DATA) + (k + 1) * 10])
        tolerance = 1e-6
        kmeans_model = Kmeansclustering(latent_TEST_DATA, K=k, max_iters=150, tol=tolerance)
        cluster_labels = kmeans_model.predict(latent_TEST_DATA)
        avg_silhouette_score = silhouette_score(latent_TEST_DATA, cluster_labels)
        print('Average silhouette coefficient (ASC) score for {k} clusters:', avg_silhouette_score)
        sample_silhouette_values = silhouette_samples(latent_TEST_DATA, cluster_labels)

        y_lower = 10
        for cluster_id in range(k):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == cluster_id]
            cluster_silhouette_values.sort()
            cluster_size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + cluster_size
            color = cm.nipy_spectral(float(cluster_id) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * cluster_size, str(cluster_id))
            y_lower = y_upper + 10

        ax.set_title("The silhouette plot for various cluster number")
        ax.set_xlabel("The SC values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=avg_silhouette_score, color="red", linestyle="--")
        ax.set_yticks([])
        ax.set_xticks(np.linspace(-0.3, 1.0, num=10))
        plt.suptitle('Silhouette analysis for clustering with %d cluster' % k, fontsize=16)
    plt.show()

def evaluate_dunn_index(latent_TEST_DATA, no_clusters):
    for k in no_clusters:
        tolerance = 1e-6
        model = Kmeansclustering(latent_TEST_DATA, K=k, max_iters=150, tol=tolerance)
        c_labels = model.predict(latent_TEST_DATA)
        pairwise_dist = pairwise_distances(latent_TEST_DATA)
        d_index = dunn(pairwise_dist, c_labels)
        print('Dunn index for the cluster number', k, 'is', d_index)