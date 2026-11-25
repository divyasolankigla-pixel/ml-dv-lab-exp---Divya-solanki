"""Experiment 5: Geographical Coordinate Bubble Visualization."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_city_data() -> pd.DataFrame:
    """Create a dataset of major cities with coordinates and population (millions)."""
    return pd.DataFrame(
        {
            "City": [
                "New Delhi",
                "Mumbai",
                "Bengaluru",
                "Kolkata",
                "Chennai",
                "Hyderabad",
                "Ahmedabad",
                "Jaipur",
            ],
            "Latitude": [28.61, 19.08, 12.97, 22.57, 13.08, 17.39, 23.02, 26.91],
            "Longitude": [77.21, 72.88, 77.59, 88.36, 80.27, 78.49, 72.57, 75.79],
            "Population_M": [32.9, 21.3, 13.6, 15.3, 11.9, 10.8, 8.6, 4.2],
            "Region": [
                "North",
                "West",
                "South",
                "East",
                "South",
                "South",
                "West",
                "North",
            ],
        }
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = build_city_data()
    print("=== Experiment 5: Geographical City Coordinates ===")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="Longitude",
        y="Latitude",
        hue="Region",
        size="Population_M",
        sizes=(120, 1200),
        alpha=0.78,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"{row['City']} ({row['Population_M']:.1f}M)",
            (row["Longitude"], row["Latitude"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
            fontweight="medium",
        )

    ax.set_title("Geographical Distribution of Major Cities by Population", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(70, 91)
    ax.set_ylim(10, 31)
    ax.legend(title="Region / Pop (M)", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
