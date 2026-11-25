"""Experiment 3: Bar and Line Charts for Monthly Product Sales."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_monthly_sales() -> pd.DataFrame:
    """Create a reproducible monthly sales dataset across three products."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    data = {
        "Month": months * 3,
        "Product": ["Laptops"] * 6 + ["Tablets"] * 6 + ["Headphones"] * 6,
        "Sales": [
            42, 48, 55, 51, 63, 70,
            28, 31, 35, 40, 38, 45,
            19, 24, 27, 30, 34, 41,
        ],
    }
    df = pd.DataFrame(data)
    df["Month"] = pd.Categorical(df["Month"], categories=months, ordered=True)
    return df


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = build_monthly_sales()

    totals = df.groupby("Product", as_index=False)["Sales"].sum()
    print("=== Experiment 3: Half-Year Product Totals ===")
    print(totals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=df, x="Month", y="Sales", hue="Product", palette="Set2", ax=axes[0])
    axes[0].set_title("Monthly Sales Comparison (Grouped Bar Chart)")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Units Sold (Thousands)")

    sns.lineplot(
        data=df,
        x="Month",
        y="Sales",
        hue="Product",
        style="Product",
        markers=True,
        dashes=False,
        linewidth=2.3,
        ax=axes[1],
    )
    axes[1].set_title("Monthly Sales Trend (Line Chart)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Units Sold (Thousands)")

    fig.suptitle("Bar and Line Charts: Product Sales Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
