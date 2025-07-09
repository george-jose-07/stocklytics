# 📈 Stocklytics - Advanced Stock Price Forecasting Platform

**Stocklytics** is a comprehensive stock price forecasting and prediction platform built with Python and Streamlit. It leverages multiple machine learning and deep learning models to provide accurate stock price predictions with interactive visualizations.

**Link:** 
```bash
https://stocklytics.streamlit.app/
```

## 🚀 Features

### 📊 Multiple Model Support
- **Classical Models**: ARIMA, SARIMA
- **Deep Learning Models**: LSTM, CNN, RNN, GRU
- **Hybrid Models**: LSTM-CNN-RNN, LSTM-GRU
- **Time Series Models**: Prophet

### 📈 Data Sources
- **CSV Upload**: Upload your own stock data with OHLC (Open, High, Low, Close) prices and volume
- **Yahoo Finance Integration**: Real-time stock data fetching using yfinance
- **Date Range Selection**: Flexible date range for historical data

### 🎛️ Customizable Parameters
- **Neural Network Models**: Epochs, Lookback Window, Units, Batch Size, Future Prediction Days and Test-Train Split 
- **ARIMA/SARIMA**: Manual parameter setting or automatic optimization with RMSE-based selection
- **Prophet**: Tuning modes (Quick/Comprehensive), Frequency settings (Business/Daily), Uncertainty samples

### 📊 Interactive Visualizations
- **Plotly Charts**: Interactive training, testing, and forecast visualizations
- **Residual Analysis**: Model performance evaluation with residual plots
- **Error Metrics**: MAE, RMSE, MAPE, and other statistical measures

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
Clone the repository
```bash
git clone https://github.com/george-jose-07/stocklytics.git
cd stocklytics
```
Install required packages
```bash
pip install -r requirements.txt
```
Run the application
```bash
streamlit run streamlit_app.py
```

### Required Dependencies
```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
yfinance>=0.1.70
scikit-learn>=1.0.0
tensorflow>=2.8.0
plotly>=5.0.0
statsmodels>=0.13.0
prophet>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 📱 Usage

### 1. Launch the Application
```bash
streamlit run streamlit_app.py
```

### 2. Select Data Source
- **Upload CSV**: Upload a CSV file with columns: Date, Open, High, Low, Close, Volume
- **Yahoo Finance**: Enter stock ticker symbol (e.g., AAPL, GOOGL, TSLA)

### 3. Choose Model
Select from available models:
- ARIMA
- SARIMA
- LSTM
- CNN
- RNN
- GRU
- Prophet
- LSTM-CNN-RNN (Hybrid)
- LSTM-GRU (Hybrid)

### 4. Set Train-Test Split
Configure the percentage split for training and testing data.

### 5. Configure Parameters

#### For Neural Network Models (LSTM, CNN, RNN, GRU, Hybrids):
- **Epochs**: Number of training iterations
- **Lookback Window**: Number of previous days to consider
- **Units**: Number of neurons in hidden layers
- **Batch Size**: Training batch size
- **Days to Predict**: Future prediction horizon

#### For ARIMA/SARIMA:
- **Manual Mode**: Set p, d, q parameters directly
- **Auto Mode**: Provide parameter ranges for automatic optimization

#### For Prophet:
- **Tuning Mode**: Quick or Comprehensive
- **Frequency**: Business days (B) or Daily (D)
- **Uncertainty Samples**: Number of samples for uncertainty estimation

### 6. Train and Predict
Click "Train Model" to start the forecasting process and view results.

## 📊 Model Details

### Classical Time Series Models
- **ARIMA**: AutoRegressive Integrated Moving Average for trend-based forecasting
- **SARIMA**: Seasonal ARIMA for data with seasonal patterns

### Deep Learning Models
- **LSTM**: Long Short-Term Memory networks for sequential data
- **CNN**: Convolutional Neural Networks for pattern recognition
- **RNN**: Recurrent Neural Networks for time series
- **GRU**: Gated Recurrent Units for efficient sequence modeling

### Hybrid Models
- **LSTM-CNN-RNN**: Combines LSTM's memory, CNN's pattern recognition, and RNN's sequence processing
- **LSTM-GRU**: Leverages LSTM's long-term memory with GRU's efficiency

### Prophet
Facebook's Prophet model for time series forecasting with strong seasonality support.

## 📈 Performance Metrics

The platform provides comprehensive evaluation metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R-squared (R²)**

## 🖼️ Screenshots

<img src="https://github.com/user-attachments/assets/6109e19e-419d-44ff-a2f7-6c1672f87f48" width="500"/>
<img src="https://github.com/user-attachments/assets/44546d63-d771-4e48-b4f1-c4013007a482" width="500"/>
<img src="https://github.com/user-attachments/assets/0182aad5-6c83-4899-ac54-023f368787c4" width="500"/>
<img src="https://github.com/user-attachments/assets/1b9590c3-f14a-45a8-bdbc-6128ad15e2db" width="500"/>
<img src="https://github.com/user-attachments/assets/1e2ac449-4142-475d-8ced-b2db32d4b824" width="500"/>
<img src="https://github.com/user-attachments/assets/85ac89fd-4df0-4f43-8f22-5baacb15fb84" width="500"/>
<img src="https://github.com/user-attachments/assets/a4b763d9-bc46-45d7-b89c-30697569b6a8" width="500"/>
<img src="https://github.com/user-attachments/assets/f851290a-c2b4-43a1-b078-548d74a70acb" width="500"/>
<img src="https://github.com/user-attachments/assets/5b931d86-d3e7-431a-9345-757de8d9edae" width="500"/>
<img src="https://github.com/user-attachments/assets/1ce06c31-144b-4f01-99a7-926a31a8a5ac" width="500"/>
<img src="https://github.com/user-attachments/assets/a0437679-2490-4d68-b25f-36b8cdf535bc" width="500"/>
<img src="https://github.com/user-attachments/assets/ba930887-823d-4c22-9e43-481bc9eb004e" width="500"/>
<img src="https://github.com/user-attachments/assets/69fcc450-12cd-4967-a0e7-d754903edcbf" width="500"/>
<img src="https://github.com/user-attachments/assets/13e0255d-9ee3-4f03-ad18-cc68be295ca6" width="500"/>

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **Yahoo Finance** for providing free stock data
- **TensorFlow/Keras** for deep learning capabilities
- **Prophet** for time series forecasting
- **Plotly** for interactive visualizations

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
