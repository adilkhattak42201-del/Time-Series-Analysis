{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b1a2c3d4-0001-4a11-8899-112233445501",
   "metadata": {},
   "source": [
    "## ⏭️ Step 22: Lag Features, Shift & Exponential Moving Average (EWM)\n",
    "\n",
    "**Summary:** Time series forecasting mein aksar hume pichle din ki sales (`lag`) ya recent dino ko zyada weight dene wala average (`ewm`) chahiye hota hai. Is step mein hum yeh seekhenge."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b1a2c3d4-0002-4a11-8899-112233445502",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pichle din ki sales ko nayi column 'Prev_Day_Sales' mein laane ke liye .shift(1) use karte hain\n",
    "df_cleaned['Prev_Day_Sales'] = df_cleaned['Sales'].shift(1)\n",
    "\n",
    "# 7-day Rolling Mean aur Exponentially Weighted Moving Average (span=7) nikaal rahe hain\n",
    "df_cleaned['Rolling_7D_Mean'] = df_cleaned['Sales'].rolling(window=7).mean()\n",
    "df_cleaned['EWM_7D'] = df_cleaned['Sales'].ewm(span=7, adjust=False).mean()\n",
    "\n",
    "# Pehli 10 rows dekh kar farq samajhte hain\n",
    "df_cleaned.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b1a2c3d4-0003-4a11-8899-112233445503",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Actual Sales, 7-Day Rolling Mean aur EWM ko ek hi graph par compare karte hain\n",
    "plt.figure(figsize=(11, 5))\n",
    "plt.plot(df_cleaned.index, df_cleaned['Sales'], label='Daily Sales', marker='o', alpha=0.6)\n",
    "plt.plot(df_cleaned.index, df_cleaned['Rolling_7D_Mean'], label='7-Day Rolling Mean', linewidth=2)\n",
    "plt.plot(df_cleaned.index, df_cleaned['EWM_7D'], label='7-Day EWM (Smooth)', linewidth=2, linestyle='--')\n",
    "\n",
    "plt.title('July 2026 - Sales vs Rolling Mean vs EWM', fontsize=14)\n",
    "plt.xlabel('Date', fontsize=12)\n",
    "plt.ylabel('Sales (in USD)', fontsize=12)\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1a2c3d4-0004-4a11-8899-112233445504",
   "metadata": {},
   "source": [
    "## 🧩 Step 23: Time Series Decomposition (Trend, Seasonality & Residuals)\n",
    "\n",
    "**Summary:** Kisi bhi time series data ko 3 hisson mein tora ja sakta hai: **Trend** (lambay arsay ka rukh), **Seasonality** (repeat hone wala pattern, jaise hafta ya mahina), aur **Residual/Noise** (baki bacha random utar-charhao)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b1a2c3d4-0005-4a11-8899-112233445505",
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.tsa.seasonal import seasonal_decompose\n",
    "\n",
    "# Data ko decompose kar rahe hain (period=7 kyunki data daily hai aur weekly cycle 7 din ka hota hai)\n",
    "decomposition = seasonal_decompose(df_cleaned['Sales'], model='additive', period=7)\n",
    "\n",
    "# Teeno components ka plot banate hain\n",
    "fig = decomposition.plot()\n",
    "fig.set_size_inches(10, 7)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1a2c3d4-0006-4a11-8899-112233445506",
   "metadata": {},
   "source": [
    "## 🔮 Step 24: Train-Test Split & Simple Forecasting (Baseline)\n",
    "\n",
    "**Summary:** Forecasting model bananey se pehle data ko **Train** aur **Test** sets mein baantna zaroori hai taaki hum check kar sakein ke hamara andaza (`forecast`) kitna durust hai. Yahan hum aakhri 7 dino ki sales predict karne ke liye ek simple baseline (last 7-day average) test kar rahe hain."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "b1a2c3d4-0007-4a11-8899-112233445507",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
    "\n",
    "# Aakhri 7 din test set ke tor par alag kar rahe hain, baqi pehle wale din train set hain\n",
    "train = df_cleaned['Sales'].iloc[:-7]\n",
    "test = df_cleaned['Sales'].iloc[-7:]\n",
    "\n",
    "# Simple Baseline Forecast: Train set ke aakhri 7 dino ki average ko agle 7 dino ke liye forecast maan liya\n",
    "baseline_forecast_value = train.iloc[-7:].mean()\n",
    "forecast = pd.Series([baseline_forecast_value] * len(test), index=test.index)\n",
    "\n",
    "# Error check karte hain (MAE aur RMSE jitna kam ho, utna behtar)\n",
    "mae = mean_absolute_error(test, forecast)\n",
    "rmse = np.sqrt(mean_squared_error(test, forecast))\n",
    "\n",
    "print(f\"Baseline Forecast Value: {baseline_forecast_value:.2f}\")\n",
    "print(f\"MAE  (Mean Absolute Error): {mae:.2f}\")\n",
    "print(f\"RMSE (Root Mean Squared Error): {rmse:.2f}\")\n",
    "\n",
    "# Actual Test Sales vs Forecast ka comparison\n",
    "comparison_df = pd.DataFrame({'Actual_Sales': test, 'Forecast_Sales': forecast})\n",
    "comparison_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "b1a2c3d4-0008-4a11-8899-112233445508",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train, Actual Test aur Forecast ko graph par visualize karte hain\n",
    "plt.figure(figsize=(11, 5))\n",
    "plt.plot(train.index, train, label='Train Sales', marker='o')\n",
    "plt.plot(test.index, test, label='Actual Test Sales', marker='o', color='green')\n",
    "plt.plot(forecast.index, forecast, label='Forecast (Baseline)', marker='x', linestyle='--', color='red')\n",
    "\n",
    "plt.title('Sales Forecast Evaluation (Last 7 Days of July 2026)', fontsize=14)\n",
    "plt.xlabel('Date', fontsize=12)\n",
    "plt.ylabel('Sales (in USD)', fontsize=12)\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
