# 📈 Stock Market Analysis & Prediction

An interactive **Streamlit** application for analyzing and forecasting stock prices. Upload your stock CSV file, explore insightful visualizations, and leverage advanced forecasting models including ARIMA, SARIMA, LSTM, and Prophet.

---

## 🚀 Features

### 🔹 1. Main Page: Data Upload & Visualization

- **CSV File Upload** — Accepts stock data with columns:  
  - `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- **Visualizations:**  
  - 📉 **Stock Closing Price Over Time**  
  - 📊 **OHLC Prices Over Time**  
  - 📈 **Trading Volume Over Time**  
  - 📌 **Basic Statistics** summary table

---

### 🔹 2. ARIMA Forecasting

- **Train-Test Split Configuration** — Interactive split for time series.
- **Stationarity Analysis** — Visualize and test stationarity.
- **Model Configuration:**  
  - Pass ARIMA parameters (`p`, `d`, `q`).
- **Training & Forecasting:**  
  - Train model at the click of a button.
- **Results & Analysis:**  
  - Forecasting Results  
  - 📊 Forecast vs Actual Graph  
  - 📉 Residuals Analysis  
  - 🔍 Additional Analysis & Model Interpretation

---

### 🔹 3. SARIMA Forecasting

- **Model Configuration:**  
  - Pass SARIMA parameters (`p`, `d`, `q`), seasonal parameters, and ARIMA parameters.
- **Model Training:**  
  - Run SARIMA model forecasting.
- **Results & Diagnostics:**  
  - RMSE, AIC, BIC  
  - 📊 Forecast Results Graph  
  - 📉 Residual Analysis Graph  
  - 🩺 Model Diagnostics  
  - 📋 Detailed Model Summary  
  - 📈 Forecast Statistics

---

### 🔹 4. LSTM Forecasting

- **Model Configuration:**  
  - Pass LSTM parameters (e.g., epochs, units, batch size).
- **Model Training & Forecasting:**  
  - Run LSTM forecast.
- **Results & Analysis:**  
  - RMSE  
  - Training Data Points  
  - 📈 Predicted Price (Day 7)  
  - 📊 Graph with LSTM Forecast and Future Price  
  - 📅 Next Week Prediction Focus  
  - 📋 Future Price Predictions Table  
  - 📉 Model Training Loss Graph  
  - 🔍 Historical Prediction Statistics  
  - 🧩 Model Architecture

---

### 🔹 5. Prophet Forecasting

- **Model Configuration:**  
  - Pass Prophet parameters (growth, seasonality).
- **Model Training & Forecasting:**  
  - Run Prophet forecast.
- **Results & Analysis:**  
  - 📋 Best Parameters Found Table  
  - RMSE  
  - 📅 Next Week Price Predictions Table  
  - 📈 Prophet Forecast for Next Days Graph  
  - 📊 Forecast Components Analysis Graph  
  - 🩺 Model Validation Details Table  
  - 🔍 Prediction Analysis & Insights

---

## 📂 Expected CSV Format

| Date       | Open  | High  | Low   | Close | Volume  |
|------------|-------|-------|-------|-------|---------|
| 2023-01-01 | 100   | 102   | 99.5  | 101   | 1000000 |
| 2023-01-02 | 101.5 | 103   | 100.8 | 102.2 | 1200000 |
| 2023-01-03 | 99.8  | 101.2 | 99    | 100.5 | 950000  |

👉 **Note:** The CSV file must have at minimum a `Close` column for ARIMA analysis.

---


### 1️⃣ Clone the Repository

```bash
git clone https://github.com/george-jose-07/stock-analysis-and-predictions-.git
cd stock-analysis-and-predictions-
