"""Experiment 6: Pie and Donut Charts for Budget Allocation."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_budget_data() -> pd.DataFrame:
    """Create a sample annual department budget allocation dataset."""
    return pd.DataFrame(
        {
            "Category": [
                "Engineering",
                "Marketing",
                "Operations",
                "Research",
                "Support",
                "Admin",
            ],
            "Budget_K": [420, 210, 180, 160, 130, 100],
        }
    )


def main() -> None:
    sns.set_theme(style="white")
    df = build_budget_data()
    df["Share_Pct"] = (df["Budget_K"] / df["Budget_K"].sum() * 100).round(1)
    print("=== Experiment 6: Budget Allocation Summary ===")
    print(df.to_string(index=False))

    colors = sns.color_palette("pastel", len(df))
    explode = [0.07 if i == 0 else 0.0 for i in range(len(df))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].pie(
        df["Budget_K"],
        labels=df["Category"],
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[0].set_title("Budget Breakdown (Pie Chart)")

    axes[1].pie(
        df["Budget_K"],
        labels=df["Category"],
        autopct="%1.1f%%",
        pctdistance=0.78,
        startangle=140,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.5},
    )
    total_budget = df["Budget_K"].sum()
    axes[1].text(
        0,
        0,
        f"Total\n${total_budget:,}K",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_title("Budget Breakdown (Donut Chart)")

    fig.suptitle("Pie and Donut Charts: Department Budget Share", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
