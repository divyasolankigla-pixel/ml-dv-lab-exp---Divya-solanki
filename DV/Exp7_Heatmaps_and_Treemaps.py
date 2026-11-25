"""Experiment 7: Correlation Heatmap and Proportional Treemap."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_store_metrics() -> pd.DataFrame:
    """Create store performance metrics for correlation analysis."""
    rng = np.random.default_rng(7)
    ad_spend = rng.uniform(10, 60, 40)
    foot_traffic = 120 + 4.5 * ad_spend + rng.normal(0, 25, 40)
    revenue = 500 + 18 * foot_traffic + rng.normal(0, 400, 40)
    discount_rate = np.clip(22 - 0.18 * ad_spend + rng.normal(0, 2.5, 40), 2, 25)
    customer_rating = np.clip(3.2 + 0.02 * ad_spend + rng.normal(0, 0.25, 40), 1, 5)

    return pd.DataFrame(
        {
            "Ad_Spend": ad_spend,
            "Foot_Traffic": foot_traffic,
            "Revenue": revenue,
            "Discount_Pct": discount_rate,
            "Rating": customer_rating,
        }
    )


def draw_treemap(ax, labels: list[str], values: list[float], colors) -> None:
    """Render a recursive binary-split proportional treemap using Matplotlib patches."""

    def split_and_draw(x: float, y: float, w: float, h: float, items: list[tuple[str, float, object]]) -> None:
        if not items:
            return
        if len(items) == 1:
            label, val, color = items[0]
            rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(
                x + w / 2,
                y + h / 2,
                f"{label}\n${val:.0f}K",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="#1b1b1b",
            )
            return

        total = sum(v for _, v, _ in items)
        half = total / 2.0
        running = 0.0
        split_idx = 1
        for i, (_, val, _) in enumerate(items[:-1], start=1):
            running += val
            split_idx = i
            if running >= half:
                break

        first = items[:split_idx]
        second = items[split_idx:]
        ratio = sum(v for _, v, _ in first) / total

        if w >= h:
            w1 = w * ratio
            split_and_draw(x, y, w1, h, first)
            split_and_draw(x + w1, y, w - w1, h, second)
        else:
            h1 = h * ratio
            split_and_draw(x, y, w, h1, first)
            split_and_draw(x, y + h1, w, h - h1, second)

    sorted_items = sorted(zip(labels, values, colors), key=lambda item: item[1], reverse=True)
    split_and_draw(0.0, 0.0, 100.0, 100.0, sorted_items)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


def main() -> None:
    sns.set_theme(style="white")
    metrics_df = build_store_metrics()
    corr = metrics_df.corr()

    categories = ["Electronics", "Apparel", "Home Decor", "Groceries", "Sports", "Books"]
    rev_k = [480, 320, 260, 210, 150, 90]

    print("=== Experiment 7: Feature Correlation Matrix ===")
    print(corr.round(2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0],
    )
    axes[0].set_title("Store Metrics Correlation Heatmap")

    palette = sns.color_palette("Set3", len(categories))
    draw_treemap(axes[1], categories, rev_k, palette)
    axes[1].set_title("Category Revenue Share (Treemap)")

    fig.suptitle("Heatmaps and Treemaps: Retail Analytics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
