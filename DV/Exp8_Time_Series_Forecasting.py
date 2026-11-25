"""Experiment 8: Time Series Decomposition Trend and 6-Month Forecasting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_sales_series(months: int = 30) -> pd.DataFrame:
    """Create a synthetic monthly sales time series with trend and annual seasonality."""
    rng = np.random.default_rng(12)
    dates = pd.date_range(start="2024-01-01", periods=months, freq="MS")
    t = np.arange(months)
    trend = 200.0 + 4.5 * t
    seasonal = 30.0 * np.sin(2.0 * np.pi * t / 12.0)
    noise = rng.normal(0.0, 7.5, size=months)
    sales = trend + seasonal + noise
    return pd.DataFrame({"Date": dates, "t": t, "Sales": np.round(sales, 2)})


def fit_seasonal_trend(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build design matrix [1, t, sin(2*pi*t/12), cos(2*pi*t/12)] and fit OLS."""
    design = np.column_stack(
        [
            np.ones_like(t),
            t,
            np.sin(2.0 * np.pi * t / 12.0),
            np.cos(2.0 * np.pi * t / 12.0),
        ]
    )
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def predict_seasonal_trend(t: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Generate predictions from fitted seasonal trend parameters."""
    design = np.column_stack(
        [
            np.ones_like(t),
            t,
            np.sin(2.0 * np.pi * t / 12.0),
            np.cos(2.0 * np.pi * t / 12.0),
        ]
    )
    return design @ beta


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = generate_sales_series(30)

    train_df = df.iloc[:-6].copy()
    val_df = df.iloc[-6:].copy()

    beta_val = fit_seasonal_trend(train_df["t"].to_numpy(), train_df["Sales"].to_numpy())
    val_pred = predict_seasonal_trend(val_df["t"].to_numpy(), beta_val)
    mae = float(np.mean(np.abs(val_df["Sales"].to_numpy() - val_pred)))
    rmse = float(np.sqrt(np.mean((val_df["Sales"].to_numpy() - val_pred) ** 2)))

    beta_full = fit_seasonal_trend(df["t"].to_numpy(), df["Sales"].to_numpy())
    fitted = predict_seasonal_trend(df["t"].to_numpy(), beta_full)
    resid_std = float(np.std(df["Sales"].to_numpy() - fitted))

    future_t = np.arange(len(df), len(df) + 6)
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS"
    )
    forecast = predict_seasonal_trend(future_t, beta_full)

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Forecast_Sales": np.round(forecast, 2),
            "Lower_95": np.round(forecast - 1.96 * resid_std, 2),
            "Upper_95": np.round(forecast + 1.96 * resid_std, 2),
        }
    )

    print("=== Experiment 8: Time Series Forecasting ===")
    print(f"Validation Holdout (Last 6 Months) -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print("\nNext 6-Month Forecast:")
    print(forecast_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Date"], df["Sales"], marker="o", label="Historical Sales", color="#264653")
    ax.plot(df["Date"], fitted, linestyle="--", label="Seasonal Trend Fit", color="#2a9d8f")
    ax.plot(
        forecast_df["Date"],
        forecast_df["Forecast_Sales"],
        marker="s",
        linewidth=2.2,
        label="6-Month Forecast",
        color="#e76f51",
    )
    ax.fill_between(
        forecast_df["Date"],
        forecast_df["Lower_95"],
        forecast_df["Upper_95"],
        color="#e76f51",
        alpha=0.2,
        label="95% Confidence Band",
    )

    ax.set_title("Monthly Sales History & 6-Month Seasonal Trend Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Volume")
    ax.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
