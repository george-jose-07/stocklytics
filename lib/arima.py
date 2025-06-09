import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Try importing statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.title("📈 ARIMA Model Forecasting")
st.write("ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.")

# Check if data is available
if st.session_state.df is None or st.session_state.df.empty:
    st.warning("⚠️ No data uploaded. Please go back to the 'Data Upload & Visualization' page and upload a CSV file.")
    st.info("👈 Use the sidebar navigation to go back to the main page.")
    st.stop()

df = st.session_state.df.copy()

# Check if Close column exists
if 'Close' not in df.columns:
    st.error("❌ 'Close' column not found in the dataset. Please ensure your CSV has a 'Close' column.")
    st.stop()


# Check if we have datetime index
has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
if has_datetime_index:
    date_range_str = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
else:
    date_range_str = "No date information available"

# Display data info
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Data Points", len(df))
with col2:
    st.metric("Stock Name", st.session_state.stock_name)

st.write("Date Range", date_range_str)
st.write("---")

# Data preparation section
# Handle missing values
missing_values = df['Close'].isnull().sum()
if missing_values > 0:
    st.warning(f"Found {missing_values} missing values in Close price. These will be filled using forward fill method.")
    df['Close'] = df['Close'].fillna(method='ffill')

# Train-test split configuration
st.subheader("Train-Test Split Configuration")
train_percentage = st.slider("Training Data Percentage", min_value=60, max_value=90, value=80, step=5)
train_size = int(len(df) * (train_percentage / 100))

# Split the data
train_data = df['Close'][:train_size]
test_data = df['Close'][train_size:]

# Get date indices for plotting
if has_datetime_index:
    train_dates = df.index[:train_size]
    test_dates = df.index[train_size:]
    full_dates = df.index
else:
    train_dates = list(range(len(train_data)))
    test_dates = list(range(len(test_data), len(df)))
    full_dates = list(range(len(df)))

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Data Size", len(train_data))
with col2:
    st.metric("Testing Data Size", len(test_data))

# Visualize train-test split
fig_split = go.Figure()
fig_split.add_trace(go.Scatter(
    x=train_dates, 
    y=train_data, 
    mode='lines', 
    name='Training Data',
    line=dict(color='blue')
))
fig_split.add_trace(go.Scatter(
    x=test_dates, 
    y=test_data, 
    mode='lines', 
    name='Testing Data',
    line=dict(color='red')
))
fig_split.update_layout(
    title=f'{st.session_state.stock_name} - Train-Test Split Visualization',
    xaxis_title='Date' if has_datetime_index else 'Time Period',
    yaxis_title='Close Price',
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig_split, use_container_width=True)

st.write("---")

# ARIMA Model Section
st.header("🤖 ARIMA Model Training & Forecasting")

if not STATSMODELS_AVAILABLE:
    st.error("❌ statsmodels library is not installed. Please install it using: pip install statsmodels")
    st.stop()

# ARIMA Parameters
st.subheader("🔧 ARIMA Parameters")
auto_search = st.radio(
    "Parameter Selection",
    ["Manual Selection", "Automatic Selection"],
    help="Choose between manual parameter selection or automatic parameter selection."
)

# Manual parameter input
if auto_search == "Manual Selection":
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("P (AR terms)", min_value=0, max_value=10, value=1, help="Autoregressive terms")
    with col2:
        d = st.number_input("D (Differencing)", min_value=0, max_value=3, value=1, help="Degree of differencing")
    with col3:
        q = st.number_input("Q (MA terms)", min_value=0, max_value=10, value=1, help="Moving average terms")


# Train ARIMA Model
if st.button("🚀 Train ARIMA Model", type="primary"):
    with st.spinner("🔄 Training ARIMA Model... This may take a moment."):
        try:
            best_aic = float('inf')
            best_model = None
            best_order = None
            results_log = []

            if auto_search == "Automatic Selection":
                # Try different parameter combinations
                progress_bar = st.progress(0)
                combinations = []
                
                # More systematic parameter search
                for p_val in range(0, 4):
                    for d_val in range(0, 3):
                        for q_val in range(0, 4):
                            combinations.append((p_val, d_val, q_val))
                
                total_combinations = len(combinations)
                successful_fits = 0
                
                for i, (p_val, d_val, q_val) in enumerate(combinations):
                    try:
                        model = ARIMA(train_data, order=(p_val, d_val, q_val))
                        fitted_model = model.fit()
                        
                        results_log.append({
                            'Order': f"({p_val},{d_val},{q_val})",
                            'AIC': fitted_model.aic,
                            'BIC': fitted_model.bic
                        })
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                            best_order = (p_val, d_val, q_val)
                        
                        successful_fits += 1
                            
                        progress_bar.progress((i + 1) / total_combinations)
                    except Exception as e:
                        results_log.append({
                            'Order': f"({p_val},{d_val},{q_val})",
                            'AIC': 'Failed',
                            'BIC': 'Failed'
                        })
                        continue
                
                progress_bar.empty()
                
                if best_model is None:
                    raise Exception("No suitable ARIMA model found. Try adjusting the parameter ranges.")
                    
                st.subheader(f"🎯 Best ARIMA parameters found: ARIMA{best_order}")                
                
            else:
                # Use user-specified parameters
                model = ARIMA(train_data, order=(p, d, q))
                best_model = model.fit()
                best_order = (p, d, q)
                best_aic = best_model.aic
            

            
            
            # Immediately proceed with forecasting after successful model training
            st.write("---")
            st.header("📈 Forecasting Results")
            
            try:
                # Generate forecasts
                forecast_result = best_model.forecast(steps=len(test_data))
                arima_forecast = forecast_result
                
                # Get confidence intervals if available
                forecast_ci = None
                try:
                    forecast_ci = best_model.get_forecast(steps=len(test_data)).conf_int()
                except:
                    pass
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
                mae = mean_absolute_error(test_data, arima_forecast)
                
                
                # Display metrics
                st.subheader("📊 Forecast Accuracy Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
                with col2:
                    st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                
                # Forecast visualization
                st.subheader("📈 Forecast vs Actual")
                
                # Create comprehensive plot
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Full Time Series with Forecast', 'Forecast Period (Zoomed)'),
                    vertical_spacing=0.2,
                    row_heights=[1, 0.5]
                )
                
                # Full time series plot
                fig.add_trace(go.Scatter(
                    x=train_dates, 
                    y=train_data, 
                    mode='lines', 
                    name='Training Data',
                    line=dict(color='blue', width=1.5),
                    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_data, 
                    mode='lines', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=arima_forecast, 
                    mode='lines', 
                    name='ARIMA Forecast',
                    line=dict(color='green', width=2),
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                # Zoomed forecast period
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_data, 
                    mode='lines+markers', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=arima_forecast, 
                    mode='lines+markers', 
                    name='ARIMA Forecast',
                    line=dict(color='green', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=2, col=1)
                
                fig.update_layout(
                    title=f'{st.session_state.stock_name} - ARIMA{best_order} Forecast Results',
                    height=1000,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                fig.update_xaxes(title_text="Date" if has_datetime_index else "Time Period", row=1, col=1)
                fig.update_xaxes(title_text="Date" if has_datetime_index else "Test Period", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Price ($)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                residuals = test_data.values - arima_forecast
                # Additional metrics
                st.subheader("📊 Additional Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
                    st.metric("Std Residual", f"{np.std(residuals):.4f}")
                
                with col2:
                    st.metric("Min Error", f"{np.min(residuals):.2f}")
                    st.metric("Max Error", f"{np.max(residuals):.2f}")
                
                # Forecast table
                st.subheader("📋 Detailed Forecast Results")
                
                # Create detailed results dataframe
                if has_datetime_index:
                    forecast_df = pd.DataFrame({
                        'Date': test_dates,
                        'Actual': test_data.values,
                        'Forecast': arima_forecast,
                        'Residual': residuals,
                        'Absolute_Error': np.abs(residuals),
                        'Percentage_Error': (residuals / test_data.values) * 100
                    })
                else:
                    forecast_df = pd.DataFrame({
                        'Period': test_dates,
                        'Actual': test_data.values,
                        'Forecast': arima_forecast,
                        'Residual': residuals,
                        'Absolute_Error': np.abs(residuals),
                        'Percentage_Error': (residuals / test_data.values) * 100
                    })
                
                # Format the dataframe for better display
                forecast_df['Actual'] = forecast_df['Actual'].round(2)
                forecast_df['Forecast'] = forecast_df['Forecast'].round(2)
                forecast_df['Residual'] = forecast_df['Residual'].round(4)
                forecast_df['Absolute_Error'] = forecast_df['Absolute_Error'].round(4)
                forecast_df['Percentage_Error'] = forecast_df['Percentage_Error'].round(2)
                
                st.dataframe(forecast_df, use_container_width=True)
                
                
                # Model interpretation
                st.subheader("💡 Model Interpretation")
                interpretation_text = f"""
                **ARIMA{best_order} Model Components:**
                - **AR({best_order[0]})**: Uses {best_order[0]} previous values to predict the next value
                - **I({best_order[1]})**: Data was differenced {best_order[1]} time(s) to make it stationary
                - **MA({best_order[2]})**: Uses {best_order[2]} previous forecast error(s) in the prediction
                
                **Performance Summary:**
                - RMSE of {rmse:.2f} indicates the average prediction error
                - MAE of {mae:.2f} shows the average absolute error
                """
                st.markdown(interpretation_text)
                
            except Exception as e:
                st.error(f"❌ Error during forecasting: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error during ARIMA model training: {str(e)}")
            st.info("💡 Try adjusting the parameters or check your data for any issues.")
            st.info("Common issues: insufficient data, too many parameters, or non-numeric data.")


with st.expander("ℹ️ About ARIMA Models"):
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)** models are used for time series forecasting.
    
    **Components:**
    - **AR (p)**: Autoregressive component - uses past values to predict future values
    - **I (d)**: Integrated component - degree of differencing to make data stationary
    - **MA (q)**: Moving average component - uses past forecast errors in prediction
    
    **When to use ARIMA:**
    - Time series data with trends
    - Data that can be made stationary through differencing
    - When you need interpretable results
    - Medium-term forecasting (not too far into the future)
    
    **Limitations:**
    - Assumes linear relationships
    - May not capture complex seasonal patterns (consider SARIMA for seasonality)
    - Performance degrades for long-term forecasts
    """)