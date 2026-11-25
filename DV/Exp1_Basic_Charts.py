"""Experiment 1: Basic Charts (Scatter Plot, Histogram, and Box Plot)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_sample_data() -> pd.DataFrame:
    """Create a reproducible student performance dataset."""
    rng = np.random.default_rng(42)
    departments = np.repeat(["CS", "ECE", "ME"], 30)
    study_hours = rng.uniform(1.5, 9.5, size=90)
    noise = rng.normal(0, 6.0, size=90)
    dept_offset = np.where(departments == "CS", 5.0, np.where(departments == "ECE", 2.0, 0.0))
    exam_score = np.clip(40 + 5.5 * study_hours + dept_offset + noise, 35, 100)

    return pd.DataFrame(
        {
            "Department": departments,
            "Study_Hours": np.round(study_hours, 2),
            "Exam_Score": np.round(exam_score, 1),
        }
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = build_sample_data()
    print("=== Experiment 1: Basic Charts Summary ===")
    print(df.groupby("Department")[["Study_Hours", "Exam_Score"]].mean().round(2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.scatterplot(
        data=df,
        x="Study_Hours",
        y="Exam_Score",
        hue="Department",
        style="Department",
        s=70,
        ax=axes[0],
    )
    axes[0].set_title("Study Hours vs. Exam Score")
    axes[0].set_xlabel("Weekly Study Hours")
    axes[0].set_ylabel("Exam Score")

    sns.histplot(data=df, x="Exam_Score", bins=12, kde=True, color="#2a9d8f", ax=axes[1])
    axes[1].set_title("Exam Score Distribution")
    axes[1].set_xlabel("Exam Score")
    axes[1].set_ylabel("Frequency")

    sns.boxplot(
        data=df,
        x="Department",
        y="Exam_Score",
        hue="Department",
        palette="Set2",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("Score Spread by Department")
    axes[2].set_xlabel("Department")
    axes[2].set_ylabel("Exam Score")

    fig.suptitle("Basic Charts: Student Performance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
