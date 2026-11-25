"""Experiment 1: Simple Linear Regression using the scikit-learn Diabetes dataset."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    diabetes = load_diabetes(as_frame=True)
    # Use Body Mass Index ('bmi') as the single predictor variable.
    X = diabetes.data[["bmi"]]
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== Experiment 1: Simple Linear Regression (Diabetes - BMI) ===")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Slope (bmi)      : {model.coef_[0]:.4f}")
    print(f"Intercept        : {model.intercept_:.4f}")
    print(f"MAE              : {mae:.4f}")
    print(f"MSE              : {mse:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"R2 Score         : {r2:.4f}")


if __name__ == "__main__":
    main()
