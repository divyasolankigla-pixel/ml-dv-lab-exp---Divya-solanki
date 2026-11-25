"""Experiment 9: K-Means Clustering on the standardized Iris dataset."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    iris = load_iris()
    X_scaled = StandardScaler().fit_transform(iris.data)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)
    ari_score = adjusted_rand_score(iris.target, labels)
    unique, counts = np.unique(labels, return_counts=True)

    print("=== Experiment 9: K-Means Clustering (Iris) ===")
    print(f"Inertia (WCSS)     : {kmeans.inertia_:.4f}")
    print(f"Silhouette Score   : {sil_score:.4f}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")

    print("\nCluster Sizes:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples")

    print("\nCluster Centers (Standardized Feature Space):")
    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        coords = ", ".join(f"{val:+.3f}" for val in center)
        print(f"  Cluster {cluster_id}: [{coords}]")


if __name__ == "__main__":
    main()
