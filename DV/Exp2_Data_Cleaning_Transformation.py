"""Experiment 2: Data Cleaning, Transformation, and Visual Summary."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_raw_dataset() -> pd.DataFrame:
    """Create a small dataset containing duplicates, missing values, and messy strings."""
    return pd.DataFrame(
        {
            "Order_ID": [101, 102, 102, 103, 104, 105, 106, 107, 108],
            "Category": [
                " electronics ",
                "Furniture",
                "Furniture",
                "OFFICE SUPPLIES",
                None,
                "Electronics",
                "furniture ",
                "Office Supplies",
                "Electronics",
            ],
            "Units": [3, 2, 2, np.nan, 4, 1, 5, 6, np.nan],
            "Unit_Price": [250.0, 420.0, 420.0, 35.0, 110.0, np.nan, 310.0, 45.0, 290.0],
            "Discount_Pct": [10, 5, 5, 0, 15, 10, np.nan, 5, 10],
        }
    )


def clean_and_transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, standardize text, impute missing values, and derive revenue columns."""
    df = raw_df.drop_duplicates(subset=["Order_ID"]).copy()
    df["Category"] = df["Category"].fillna("Unknown").str.strip().str.title()
    df["Units"] = df["Units"].fillna(df["Units"].median())
    df["Unit_Price"] = df["Unit_Price"].fillna(
        df.groupby("Category")["Unit_Price"].transform("median")
    )
    df["Unit_Price"] = df["Unit_Price"].fillna(df["Unit_Price"].median())
    df["Discount_Pct"] = df["Discount_Pct"].fillna(0.0)

    df["Gross_Amount"] = df["Units"] * df["Unit_Price"]
    df["Net_Revenue"] = df["Gross_Amount"] * (1.0 - df["Discount_Pct"] / 100.0)
    return df


def main() -> None:
    sns.set_theme(style="whitegrid")
    raw_df = build_raw_dataset()
    clean_df = clean_and_transform(raw_df)

    print("=== Raw Dataset ===")
    print(raw_df)
    print("\n=== Cleaned & Transformed Dataset ===")
    print(clean_df)

    summary = (
        clean_df.groupby("Category", as_index=False)[["Units", "Net_Revenue"]]
        .sum()
        .sort_values("Net_Revenue", ascending=False)
    )
    print("\n=== Category Summary ===")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.barplot(
        data=summary,
        x="Category",
        y="Net_Revenue",
        hue="Category",
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Net Revenue by Cleaned Category")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Net Revenue ($)")
    axes[0].tick_params(axis="x", rotation=15)

    sns.scatterplot(
        data=clean_df,
        x="Units",
        y="Net_Revenue",
        hue="Category",
        size="Unit_Price",
        sizes=(60, 240),
        ax=axes[1],
    )
    axes[1].set_title("Units Sold vs. Net Revenue")
    axes[1].set_xlabel("Units")
    axes[1].set_ylabel("Net Revenue ($)")

    fig.suptitle("Data Cleaning and Transformation Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
