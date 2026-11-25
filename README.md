# July 2026 Sales Time Series Analysis & Forecasting

A Python Jupyter Notebook covering data cleaning, exploratory data analysis (EDA), rolling/exponential smoothing, seasonal decomposition, and baseline forecasting on daily sales data (`df_cleaned` / `Sales` in USD).

## What’s Included in the Notebook

* **Data Cleaning & Feature Engineering**
  * Preparing `df_cleaned` and handling missing/duplicate entries
  * Lag features (`Prev_Day_Sales` using `.shift(1)`)
  * Moving averages (`Rolling_7D_Mean` and `EWM_7D` with `span=7`)
* **Exploratory Data Analysis & Visualization**
  * Daily sales trend plots
  * Comparison of raw daily sales, 7-day rolling mean, and 7-day exponentially weighted moving average
* **Time Series Decomposition**
  * Additive decomposition (`period=7`) into **Trend**, **Seasonality**, and **Residuals** using `statsmodels`
* **Train-Test Split & Baseline Forecasting**
  * Holding out the last 7 days as the test set
  * Computing a 7-day average baseline forecast and evaluating performance using **MAE** and **RMSE**

## Tech Stack & Dependencies

* Python 3.10+
* `pandas`, `numpy`
* `matplotlib`
* `statsmodels`
* `scikit-learn`

Install the required packages with:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn jupyter
