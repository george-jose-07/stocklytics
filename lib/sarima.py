import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

st.title("🔄 SARIMA Stock Price Forecasting")
st.write("Seasonal AutoRegressive Integrated Moving Average for time series prediction with seasonality")

# Check if data is available
if st.session_state.df is None:
    st.warning("⚠️ Please upload data first from the 'Data Upload & Visualization' page")
    st.stop()

df = st.session_state.df.copy()
stock_name = st.session_state.stock_name

# Sidebar for parameters
st.sidebar.header("SARIMA Parameters")

train_ratio = st.sidebar.slider("Training Data Ratio", 0.6, 0.9, 0.8, 0.05)

st.sidebar.subheader("Model Selection Method")
auto_search = st.sidebar.radio(
    "Parameter Selection",
    ["Manual Selection", "Grid Search"],
    help="Choose between manual parameter selection or automatic grid search"
)

if auto_search == "Manual Selection":
    st.sidebar.subheader("ARIMA Parameters (p,d,q)")
    p = st.sidebar.slider("AR order (p)", 0, 5, 1)
    d = st.sidebar.slider("Integration order (d)", 0, 3, 1)
    q = st.sidebar.slider("MA order (q)", 0, 5, 1)
    
    st.sidebar.subheader("Seasonal Parameters (P,D,Q,s)")
    P = st.sidebar.slider("Seasonal AR (P)", 0, 3, 1)
    D = st.sidebar.slider("Seasonal Integration (D)", 0, 2, 1)
    Q = st.sidebar.slider("Seasonal MA (Q)", 0, 3, 1)
    s = st.sidebar.slider("Seasonal Period (s)", 5, 50, 30, 5)
    
else:  # Grid Search
    st.sidebar.subheader("Search Ranges")
    p_range = st.sidebar.slider("AR order range (p)", 1, 3, (0, 2))
    d_range = st.sidebar.slider("Integration order range (d)", 1, 2, (0, 1))
    q_range = st.sidebar.slider("MA order range (q)", 1, 3, (0, 2))
    
    P_range = st.sidebar.slider("Seasonal AR range (P)", 1, 2, (0, 1))
    D_range = st.sidebar.slider("Seasonal Integration range (D)", 1, 2, (0, 1))
    Q_range = st.sidebar.slider("Seasonal MA range (Q)", 1, 2, (0, 1))
    s = st.sidebar.slider("Seasonal Period (s)", 5, 50, 30, 5)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    information_criterion = st.selectbox("Information Criterion", ["aic", "bic"], index=0)
    include_trend = st.selectbox("Trend Component", ["c", "ct", "ctt", "n"], 
                                index=0, help="c=constant, ct=constant+trend, ctt=constant+linear+quadratic, n=none")

def check_stationarity(timeseries, title):
    """Check stationarity using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    
    st.write(f"**{title}**")
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    
    if result[1] <= 0.05:
        st.success("✅ Series is stationary")
        return True
    else:
        st.warning("⚠️ Series is not stationary")
        return False

def grid_search_sarima(train_data, p_range, d_range, q_range, P_range, D_range, Q_range, s, criterion='aic'):
    """Perform grid search to find optimal SARIMA parameters"""
    
    best_score = float('inf')
    best_params = None
    best_seasonal_params = None
    results = []
    
    # Generate all parameter combinations
    p_values = list(range(p_range[0], p_range[1] + 1))
    d_values = list(range(d_range[0], d_range[1] + 1))
    q_values = list(range(q_range[0], q_range[1] + 1))
    P_values = list(range(P_range[0], P_range[1] + 1))
    D_values = list(range(D_range[0], D_range[1] + 1))
    Q_values = list(range(Q_range[0], Q_range[1] + 1))
    
    total_combinations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_combination = 0
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        for P, D, Q in itertools.product(P_values, D_values, Q_values):
            try:
                current_combination += 1
                progress = current_combination / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing SARIMA({p},{d},{q})({P},{D},{Q})[{s}] - {current_combination}/{total_combinations}")
                
                model = SARIMAX(train_data, 
                              order=(p, d, q),
                              seasonal_order=(P, D, Q, s),
                              trend=include_trend,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                
                fitted_model = model.fit(disp=False)
                
                score = getattr(fitted_model, criterion)
                results.append({
                    'params': (p, d, q),
                    'seasonal_params': (P, D, Q, s),
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'score': score
                })
                
                if score < best_score:
                    best_score = score
                    best_params = (p, d, q)
                    best_seasonal_params = (P, D, Q, s)
                    
            except Exception as e:
                continue
    
    progress_bar.progress(1.0)
    status_text.text("✅ Grid search completed!")
    
    return best_params, best_seasonal_params, best_score, results

if st.button("🚀 Run SARIMA Forecast", type="primary"):
    with st.spinner("Running SARIMA analysis..."):
        try:
            # Prepare data
            train_size = int(len(df) * train_ratio)
            train = df.iloc[:train_size]['Close']
            test = df.iloc[train_size:]['Close']
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Points", len(train))
            with col2:
                st.metric("Test Points", len(test))
            with col3:
                if auto_search == "Manual Selection":
                    st.metric("Seasonal Period", s)
                else:
                    st.metric("Seasonal Period", s)
            
            # Stationarity check
            st.subheader("📊 Stationarity Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                is_stationary = check_stationarity(train, "Original Series Stationarity Test")
            
            with col2:
                if not is_stationary:
                    diff_series = train.diff().dropna()
                    check_stationarity(diff_series, "First Difference Stationarity Test")
            
            # Parameter selection
            if auto_search == "Grid Search":
                st.subheader("🔍 Grid Search for Optimal Parameters")
                best_params, best_seasonal_params, best_score, search_results = grid_search_sarima(
                    train, p_range, d_range, q_range, P_range, D_range, Q_range, s, information_criterion
                )
                
                st.success(f"✅ Best parameters found!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best ARIMA Order", f"({best_params[0]},{best_params[1]},{best_params[2]})")
                    st.metric(f"Best {information_criterion.upper()}", f"{best_score:.2f}")
                with col2:
                    st.metric("Best Seasonal Order", f"({best_seasonal_params[0]},{best_seasonal_params[1]},{best_seasonal_params[2]})")
                    st.metric("Seasonal Period", f"{best_seasonal_params[3]}")
                
                # Use best parameters
                p, d, q = best_params
                P, D, Q, s = best_seasonal_params
                
            else:
                # Use manual parameters
                best_params = (p, d, q)
                best_seasonal_params = (P, D, Q, s)
            
            # Fit the final model
            st.subheader("🔧 Fitting SARIMA Model")
            with st.spinner("Fitting final SARIMA model..."):
                model = SARIMAX(train,
                              order=(p, d, q),
                              seasonal_order=(P, D, Q, s),
                              trend=include_trend,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                
                fitted_model = model.fit(disp=False)
            
            # Generate forecasts
            forecast = fitted_model.forecast(steps=len(test))
            forecast_ci = fitted_model.get_forecast(steps=len(test)).conf_int()
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test, forecast))
            
            # Display model results
            st.success(f"✅ SARIMA model fitted successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("AIC", f"{fitted_model.aic:.2f}")
            with col2:
                st.metric("BIC", f"{fitted_model.bic:.2f}")
                st.metric("Log Likelihood", f"{fitted_model.llf:.2f}")
            
            # Main forecast plot
            st.subheader("📈 Forecast Results")
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot training data
            ax.plot(train.index, train.values, label='Training Data', color='blue', alpha=0.7, linewidth=1.5)
            
            # Plot actual test data
            ax.plot(test.index, test.values, label='Actual Prices', color='red', linewidth=2)
            
            # Plot forecast
            ax.plot(test.index, forecast, label='SARIMA Forecast', 
                   color='green', linewidth=2, linestyle='--')
            
            # Plot confidence intervals
            ax.fill_between(test.index, 
                           forecast_ci.iloc[:, 0], 
                           forecast_ci.iloc[:, 1], 
                           color='green', alpha=0.2, label='95% Confidence Interval')
            
            ax.set_title(f'{stock_name} - SARIMA({p},{d},{q})({P},{D},{Q})[{s}] Forecasting Results', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add vertical line to separate train/test
            ax.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.7, linewidth=1)
            
            st.pyplot(fig)
            
            # Residual analysis
            st.subheader("📊 Residual Analysis")
            
            residuals = fitted_model.resid
            
            fig_res, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Residuals plot
            ax1.plot(train.index, residuals, color='blue', alpha=0.7)
            ax1.set_title('Residuals')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Residuals')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Residuals histogram
            ax2.hist(residuals, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            ax2.set_title('Residuals Distribution')
            ax2.set_xlabel('Residual Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot')
            ax3.grid(True, alpha=0.3)
            
            # ACF of residuals
            lags = min(20, len(residuals)//4)
            acf_vals = acf(residuals, nlags=lags, fft=False)
            ax4.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
            ax4.set_title('ACF of Residuals')
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('ACF')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add significance bounds for ACF
            n = len(residuals)
            significance_bound = 1.96 / np.sqrt(n)
            ax4.axhline(y=significance_bound, color='red', linestyle='--', alpha=0.7)
            ax4.axhline(y=-significance_bound, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig_res)
            
            # Model diagnostics
            st.subheader("🔍 Model Diagnostics")
            
            # Ljung-Box test for residual autocorrelation
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Ljung-Box Test (Residual Autocorrelation)**")
                st.dataframe(ljung_box)
                
                # Interpretation
                if (ljung_box['lb_pvalue'] > 0.05).all():
                    st.success("✅ No significant autocorrelation in residuals")
                else:
                    st.warning("⚠️ Some autocorrelation detected in residuals")
            
            with col2:
                st.write("**Model Summary Statistics**")
                st.write(f"**Model:** SARIMA({p},{d},{q})({P},{D},{Q})[{s}]")
                st.write(f"**AIC:** {fitted_model.aic:.2f}")
                st.write(f"**BIC:** {fitted_model.bic:.2f}")
                st.write(f"**HQIC:** {fitted_model.hqic:.2f}")
                st.write(f"**Log Likelihood:** {fitted_model.llf:.2f}")
            
            # Model summary and statistics
            with st.expander("🔍 Detailed Model Summary"):
                st.text(str(fitted_model.summary()))
            
            # Forecast statistics
            with st.expander("📈 Forecast Statistics"):
                forecast_df = pd.DataFrame({
                    'Date': test.index,
                    'Actual': test.values,
                    'Forecast': forecast.values,
                    'Lower CI': forecast_ci.iloc[:, 0].values,
                    'Upper CI': forecast_ci.iloc[:, 1].values,
                    'Error': test.values - forecast.values,
                    'Abs Error': np.abs(test.values - forecast.values)
                })
                
                st.dataframe(forecast_df, use_container_width=True)
                
                # Error statistics
                st.write("**Error Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"{np.mean(forecast_df['Abs Error']):.2f}")
                with col2:
                    st.metric("Mean Error", f"{np.mean(forecast_df['Error']):.2f}")
                with col3:
                    st.metric("Max Absolute Error", f"{np.max(forecast_df['Abs Error']):.2f}")
            
            # Grid search results if applicable
            if auto_search == "Grid Search":
                with st.expander("🔍 Grid Search Results"):
                    if 'search_results' in locals():
                        results_df = pd.DataFrame(search_results)
                        results_df = results_df.sort_values('score').head(10)
                        st.write("**Top 10 Parameter Combinations:**")
                        st.dataframe(results_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error during SARIMA modeling: {str(e)}")
            st.write("**Possible solutions:**")
            st.write("- Try different parameter values")
            st.write("- Check if your data has enough observations")
            st.write("- Adjust the seasonal period")
            st.write("- Try different trend components")
            st.write("- Ensure your data doesn't have missing values")

# Information about SARIMA
with st.expander("ℹ️ About SARIMA Forecasting"):
    st.write("""
    **Seasonal ARIMA (SARIMA)** extends ARIMA modeling to handle seasonal patterns in time series data.
    
    **Model Structure:** SARIMA(p,d,q)(P,D,Q)[s]
    - **(p,d,q)**: Non-seasonal parameters (AR, Integration, MA)
    - **(P,D,Q)**: Seasonal parameters
    - **[s]**: Seasonal period
    
    **Implementation Details:**
    - Uses statsmodels SARIMAX for model fitting
    - Manual parameter selection or grid search available
    - Includes stationarity testing with ADF test
    - Comprehensive residual analysis
    - Ljung-Box test for model validation
    
    **Parameters Guide:**
    - **Seasonal Period (s)**: Number of observations in one seasonal cycle
    - **Grid Search**: Automatically finds best parameters within specified ranges
    - **Manual Selection**: Allows precise control over model parameters
    - **Trend Component**: 
      - 'c': constant
      - 'ct': constant + linear trend
      - 'ctt': constant + linear + quadratic trend
      - 'n': no trend
    
    **Best Practices:**
    - Start with grid search to identify good parameter ranges
    - Check stationarity before modeling
    - Validate model with residual analysis
    - Use seasonal period appropriate for your data frequency
    """)