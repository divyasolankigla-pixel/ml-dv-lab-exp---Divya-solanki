"""Experiment 4: Interactive Dashboard with Region and Product Filters."""

import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import pandas as pd
import seaborn as sns


def build_sales_data() -> pd.DataFrame:
    """Construct a multi-region, multi-product monthly sales dataset."""
    rows = []
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    base_sales = {
        ("North", "Tea"): [120, 135, 140, 128, 150, 165],
        ("North", "Coffee"): [180, 175, 195, 205, 210, 225],
        ("North", "Juice"): [90, 95, 110, 130, 155, 170],
        ("South", "Tea"): [145, 150, 160, 155, 172, 180],
        ("South", "Coffee"): [210, 205, 220, 235, 240, 255],
        ("South", "Juice"): [115, 125, 140, 165, 185, 200],
        ("West", "Tea"): [100, 108, 115, 122, 130, 142],
        ("West", "Coffee"): [160, 168, 175, 182, 190, 202],
        ("West", "Juice"): [105, 112, 128, 148, 162, 178],
    }
    for (region, product), values in base_sales.items():
        for month, revenue in zip(months, values):
            rows.append(
                {
                    "Month": month,
                    "Region": region,
                    "Product": product,
                    "Revenue": revenue * 100,
                }
            )
    df = pd.DataFrame(rows)
    df["Month"] = pd.Categorical(df["Month"], categories=months, ordered=True)
    return df


def filter_data(df: pd.DataFrame, region: str, product: str) -> pd.DataFrame:
    """Return subset of sales data matching the chosen region and product."""
    filtered = df
    if region != "All":
        filtered = filtered[filtered["Region"] == region]
    if product != "All":
        filtered = filtered[filtered["Product"] == product]
    return filtered


def render_dashboard(
    df: pd.DataFrame,
    ax_trend,
    ax_region,
    ax_product,
    region: str,
    product: str,
) -> None:
    """Draw the filtered dashboard charts onto the given axes."""
    subset = filter_data(df, region, product)
    for ax in (ax_trend, ax_region, ax_product):
        ax.clear()

    monthly = subset.groupby("Month", observed=False, as_index=False)["Revenue"].sum()
    sns.lineplot(data=monthly, x="Month", y="Revenue", marker="o", linewidth=2.5, ax=ax_trend)
    ax_trend.set_title(f"Monthly Revenue Trend ({region} | {product})")
    ax_trend.set_ylabel("Revenue ($)")

    by_region = subset.groupby("Region", as_index=False)["Revenue"].sum()
    sns.barplot(
        data=by_region,
        x="Region",
        y="Revenue",
        hue="Region",
        palette="Blues_d",
        legend=False,
        ax=ax_region,
    )
    ax_region.set_title("Revenue by Region")
    ax_region.set_ylabel("Revenue ($)")

    by_product = subset.groupby("Product", as_index=False)["Revenue"].sum()
    sns.barplot(
        data=by_product,
        x="Product",
        y="Revenue",
        hue="Product",
        palette="Greens_d",
        legend=False,
        ax=ax_product,
    )
    ax_product.set_title("Revenue by Product")
    ax_product.set_ylabel("Revenue ($)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Sales Dashboard")
    parser.add_argument(
        "--region", default="All", choices=["All", "North", "South", "West"]
    )
    parser.add_argument(
        "--product", default="All", choices=["All", "Tea", "Coffee", "Juice"]
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    df = build_sales_data()
    initial = filter_data(df, args.region, args.product)
    print(f"=== Dashboard Filter: Region={args.region}, Product={args.product} ===")
    print(f"Records: {len(initial)} | Total Revenue: ${initial['Revenue'].sum():,.0f}")

    fig = plt.figure(figsize=(15, 8))
    ax_trend = fig.add_axes([0.24, 0.55, 0.72, 0.35])
    ax_region = fig.add_axes([0.24, 0.10, 0.33, 0.33])
    ax_product = fig.add_axes([0.63, 0.10, 0.33, 0.33])

    ax_reg_ctrl = fig.add_axes([0.03, 0.55, 0.14, 0.25], facecolor="#f4f6f8")
    ax_prod_ctrl = fig.add_axes([0.03, 0.20, 0.14, 0.25], facecolor="#f4f6f8")
    ax_reg_ctrl.set_title("Region Filter", fontsize=10, fontweight="bold")
    ax_prod_ctrl.set_title("Product Filter", fontsize=10, fontweight="bold")

    state = {"region": args.region, "product": args.product}
    reg_options = ["All", "North", "South", "West"]
    prod_options = ["All", "Tea", "Coffee", "Juice"]

    reg_radio = RadioButtons(ax_reg_ctrl, reg_options, active=reg_options.index(args.region))
    prod_radio = RadioButtons(ax_prod_ctrl, prod_options, active=prod_options.index(args.product))

    def on_change(_label: str) -> None:
        render_dashboard(df, ax_trend, ax_region, ax_product, state["region"], state["product"])
        fig.canvas.draw_idle()

    def on_region(label: str) -> None:
        state["region"] = label
        on_change(label)

    def on_product(label: str) -> None:
        state["product"] = label
        on_change(label)

    reg_radio.on_clicked(on_region)
    prod_radio.on_clicked(on_product)

    render_dashboard(df, ax_trend, ax_region, ax_product, state["region"], state["product"])
    fig.suptitle("Interactive Sales Dashboard", fontsize=15, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()
