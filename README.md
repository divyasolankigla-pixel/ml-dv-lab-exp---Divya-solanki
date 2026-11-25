# Machine Learning & Data Visualization Lab

**Name:** Divya Solanki  
**Repository:** [ml-dv-lab-exp---Divya-solanki](https://github.com/divyasolankigla-pixel/ml-dv-lab-exp---Divya-solanki)

This repository contains 9 self-contained Machine Learning experiments (using `scikit-learn` built-in datasets) and 8 self-contained Data Visualization experiments (using `pandas`, `matplotlib`, and `seaborn`), organized in chronological order inside the `ML/` and `DV/` folders.

## Repository Structure

```text
.
├── ML/
│   ├── Exp1_Simple_Linear_Regression.py
│   ├── Exp2_Multiple_Linear_Regression.py
│   ├── Exp3_Logistic_Regression.py
│   ├── Exp4_Regression_with_Regularization.py
│   ├── Exp5_Dimensionality_Reduction_PCA.py
│   ├── Exp6_SVM.py
│   ├── Exp7_Decision_Tree.py
│   ├── Exp8_Naive_Bayes.py
│   └── Exp9_KMeans_Clustering.py
├── DV/
│   ├── Exp1_Basic_Charts.py
│   ├── Exp2_Data_Cleaning_Transformation.py
│   ├── Exp3_Bar_and_Line_Charts.py
│   ├── Exp4_Interactive_Dashboard_Filters.py
│   ├── Exp5_Geo_Visualization.py
│   ├── Exp6_Pie_and_Donut_Charts.py
│   ├── Exp7_Heatmaps_and_Treemaps.py
│   └── Exp8_Time_Series_Forecasting.py
└── README.md
```

## Machine Learning Experiments (`ML/`)

1. **`Exp1_Simple_Linear_Regression.py`** — Fits a simple linear regression model on the Diabetes dataset using BMI as a single predictor and reports slope, intercept, MAE, MSE, RMSE, and $R^2$.
2. **`Exp2_Multiple_Linear_Regression.py`** — Fits a multiple linear regression model using all 10 features of the Diabetes dataset and prints feature coefficients alongside evaluation metrics.
3. **`Exp3_Logistic_Regression.py`** — Trains a standardized logistic regression pipeline on the Breast Cancer dataset and evaluates accuracy, ROC-AUC, confusion matrix, and classification report.
4. **`Exp4_Regression_with_Regularization.py`** — Compares standard Linear Regression against Ridge (L2) and Lasso (L1) models on the Diabetes dataset across multiple regularization strengths.
5. **`Exp5_Dimensionality_Reduction_PCA.py`** — Standardizes the Iris dataset and applies Principal Component Analysis (PCA) to inspect explained variance ratios and 2D component projections.
6. **`Exp6_SVM.py`** — Trains Support Vector Machine classifiers with `linear`, `rbf`, and `poly` kernels on standardized Breast Cancer data and reports the best model's metrics.
7. **`Exp7_Decision_Tree.py`** — Trains a Decision Tree classifier on the Iris dataset, reports feature importances and classification metrics, and prints the learned decision rules.
8. **`Exp8_Naive_Bayes.py`** — Applies Gaussian Naive Bayes to the Wine dataset and prints learned class priors, accuracy, confusion matrix, and classification report.
9. **`Exp9_KMeans_Clustering.py`** — Performs K-Means clustering ($k = 3$) on standardized Iris measurements and reports inertia, silhouette score, adjusted Rand index, and cluster centroids.

## Data Visualization Experiments (`DV/`)

1. **`Exp1_Basic_Charts.py`** — Generates a multi-panel figure with a scatter plot, histogram with KDE, and department-wise box plot of student performance data.
2. **`Exp2_Data_Cleaning_Transformation.py`** — Demonstrates deduplication, string standardization, missing-value imputation, revenue derivation, and visual category/scatter summaries.
3. **`Exp3_Bar_and_Line_Charts.py`** — Compares monthly product sales using a grouped bar chart and a multi-series line chart.
4. **`Exp4_Interactive_Dashboard_Filters.py`** — Renders a sales dashboard with interactive radio-button filters and optional `--region` / `--product` CLI arguments.
5. **`Exp5_Geo_Visualization.py`** — Plots major city latitude/longitude coordinates as population-scaled, region-colored bubbles with text annotations.
6. **`Exp6_Pie_and_Donut_Charts.py`** — Displays side-by-side pie and donut charts summarizing departmental budget shares.
7. **`Exp7_Heatmaps_and_Treemaps.py`** — Visualizes a store-metrics correlation heatmap alongside a proportional category-revenue treemap.
8. **`Exp8_Time_Series_Forecasting.py`** — Fits a linear-plus-seasonal trend model on monthly sales data, evaluates holdout error, and plots a 6-month forecast with a 95% confidence band.

## Running the Experiments

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run any experiment from the repository root:

```bash
python ML/Exp1_Simple_Linear_Regression.py
python DV/Exp1_Basic_Charts.py
python DV/Exp4_Interactive_Dashboard_Filters.py --region South --product Coffee
```
