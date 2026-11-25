"""Experiment 5: Dimensionality Reduction using Principal Component Analysis (PCA)."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main() -> None:
    iris = load_iris()
    X_scaled = StandardScaler().fit_transform(iris.data)

    pca_full = PCA().fit(X_scaled)
    pca_2d = PCA(n_components=2, random_state=42)
    X_reduced = pca_2d.fit_transform(X_scaled)

    print("=== Experiment 5: Dimensionality Reduction with PCA (Iris) ===")
    print(f"Original shape : {iris.data.shape}")
    print(f"Reduced shape  : {X_reduced.shape}")

    print("\nExplained Variance Ratio (All Components):")
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    for idx, (var, cum) in enumerate(
        zip(pca_full.explained_variance_ratio_, cumulative), start=1
    ):
        print(f"  PC{idx}: {var:.4f} (Cumulative: {cum:.4f})")

    print("\n2-Component Principal Axes (Loadings):")
    for idx, component in enumerate(pca_2d.components_, start=1):
        loadings = ", ".join(
            f"{feat}: {val:+.3f}" for feat, val in zip(iris.feature_names, component)
        )
        print(f"  PC{idx} -> {loadings}")

    print("\nFirst 5 Projected Samples (PC1, PC2):")
    for sample, label in zip(X_reduced[:5], iris.target[:5]):
        print(f"  [{sample[0]:+.4f}, {sample[1]:+.4f}] -> {iris.target_names[label]}")


if __name__ == "__main__":
    main()
