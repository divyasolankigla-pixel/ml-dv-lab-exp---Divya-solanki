"""Experiment 4: Regression with L1 (Lasso) and L2 (Ridge) Regularization."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_model(name: str, estimator, X_train, X_test, y_train, y_test) -> None:
    """Fit a regression pipeline and report error metrics and sparsity."""
    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    coefs = pipeline.named_steps[estimator.__class__.__name__.lower()].coef_
    nonzero = int(np.sum(np.abs(coefs) > 1e-6))

    print(
        f"{name:<22} | MSE: {mse:8.2f} | RMSE: {rmse:6.2f} | "
        f"R2: {r2:6.4f} | Non-zero coefs: {nonzero}/{len(coefs)}"
    )


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge (alpha=1.0)", Ridge(alpha=1.0, random_state=42)),
        ("Ridge (alpha=10.0)", Ridge(alpha=10.0, random_state=42)),
        ("Lasso (alpha=0.5)", Lasso(alpha=0.5, max_iter=5000, random_state=42)),
        ("Lasso (alpha=5.0)", Lasso(alpha=5.0, max_iter=5000, random_state=42)),
    ]

    print("=== Experiment 4: Regression with Regularization (Diabetes) ===")
    for name, estimator in models:
        evaluate_model(name, estimator, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
