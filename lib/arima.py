import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler


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

# Initialize and apply MinMaxScaler
scaler = MinMaxScaler()
close_prices = df['Close'].values.reshape(-1, 1)
scaled_prices = scaler.fit_transform(close_prices).flatten()

# Train-test split configuration
st.subheader("Train-Test Split Configuration")
train_percentage = st.slider("Training Data Percentage", min_value=60, max_value=90, value=80, step=5)
train_size = int(len(df) * (train_percentage / 100))

# Split the scaled data
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

# Get date indices for plotting
if has_datetime_index:
    train_dates = df.index[:train_size]
    test_dates = df.index[train_size:]
    full_dates = df.index
else:
    train_dates = list(range(len(train_data)))
    test_dates = list(range(len(train_data),len(df)))
    full_dates = list(range(len(df)))

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Data Size", len(train_data))
with col2:
    st.metric("Testing Data Size", len(test_data))

# Visualize train-test split (using original prices for better interpretation)
fig_split = go.Figure()
fig_split.add_trace(go.Scatter(
    x=train_dates, 
    y=df['Close'][:train_size], 
    mode='lines', 
    name='Training Data',
    line=dict(color='blue')
))
fig_split.add_trace(go.Scatter(
    x=test_dates, 
    y=df['Close'][train_size:], 
    mode='lines', 
    name='Testing Data',
    line=dict(color='red')
))
fig_split.update_layout(
    title=f'{st.session_state.stock_name} - Train-Test Split Visualization',
    xaxis_title='Date' if has_datetime_index else 'Time Period',
    yaxis_title='Close Price ($)',
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig_split, use_container_width=True)

st.write("---")

# ARIMA Model Section
st.header("🤖 ARIMA Model Training & Forecasting")

# ARIMA Parameters
st.subheader("🔧 ARIMA Parameters")
future_days = st.slider("Days to Predict into Future", 1, 30, 14, 1)
auto_search = st.radio(
    "Parameter Selection",
    ["Manual Selection", "Automatic Selection"],
    help="Choose between manual parameter selection or automatic parameter selection."
)

# Manual parameter input
if auto_search == "Manual Selection":
    st.subheader("Manual ARIMA Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("P (AR terms)", min_value=0, max_value=5, value=1, help="Autoregressive terms")
    with col2:
        d = st.number_input("D (Differencing)", min_value=0, max_value=2, value=1, help="Degree of differencing")
    with col3:
        q = st.number_input("Q (MA terms)", min_value=0, max_value=5, value=1, help="Moving average terms")
else:
    st.subheader("Search Ranges")
    col1, col3 = st.columns(2)
    with col1:
        p_range = st.slider("AR order range (p)", 1, 5, (0, 2))
    with col3:
        q_range = st.slider("MA order range (q)", 1, 5, (0, 2))

d_range = (0, 2)
# Train ARIMA Model
if st.button("🚀 Train ARIMA Model", type="primary"):
    with st.spinner("🔄 Training ARIMA Model... This may take a moment."):
        try:
            best_rmse = float('inf')
            best_model = None
            best_order = None
            results_log = []

            if auto_search == "Automatic Selection":
                # Try different parameter combinations
                progress_bar = st.progress(0)
                status_text = st.empty()
                current_combination = 0
                combinations = []
                
                # More systematic parameter search
                for p_val in range(p_range[0], p_range[1] + 1):
                    for d_val in range(d_range[0], d_range[1] + 1):
                        for q_val in range(q_range[0], q_range[1] + 1):
                            combinations.append((p_val, d_val, q_val))
                
                total_combinations = len(combinations)
                successful_fits = 0
                
                for i, (p_val, d_val, q_val) in enumerate(combinations):
                    current_combination += 1
                    progress = current_combination / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing ARIMA({p_val},{d_val},{q_val}) - {current_combination}/{total_combinations}")
                    try:
                        model = ARIMA(train_data, order=(p_val, d_val, q_val))
                        fitted_model = model.fit()
                        
                        # Make prediction on test data for RMSE calculation
                        forecast_result = fitted_model.forecast(steps=len(test_data))
                        rmse = np.sqrt(mean_squared_error(test_data, forecast_result))
                        
                        results_log.append({
                            'Order': f"({p_val},{d_val},{q_val})",
                            'RMSE': rmse,
                            'AIC': fitted_model.aic,
                            'BIC': fitted_model.bic
                        })
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_model = fitted_model
                            best_order = (p_val, d_val, q_val)
                        
                        successful_fits += 1
                            
                    except Exception as e:
                        results_log.append({
                            'Order': f"({p_val},{d_val},{q_val})",
                            'RMSE': 'Failed',
                            'AIC': 'Failed',
                            'BIC': 'Failed'
                        })
                        continue
                
                progress_bar.empty()
                status_text.text("✅ search completed!")
                
                if best_model is None:
                    raise Exception("No suitable ARIMA model found. Try adjusting the parameter ranges.")
                    
                st.subheader(f"🎯 Best ARIMA parameters found: ARIMA{best_order}")                
                
            else:
                # Use user-specified parameters
                model = ARIMA(train_data, order=(p, d, q))
                best_model = model.fit()
                best_order = (p, d, q)
                # Calculate RMSE for the user-specified model
                forecast_result = best_model.forecast(steps=len(test_data))
                best_rmse = np.sqrt(mean_squared_error(test_data, forecast_result))
            
            # # Store the trained model in session state for reuse
            # st.session_state.arima_model = best_model
            # st.session_state.arima_scaler = scaler
            # st.session_state.arima_order = best_order
            
            # Immediately proceed with forecasting after successful model training
            st.write("---")
            st.header("📈 Forecasting Results")
            
            try:
                # Generate forecasts for test period
                test_forecast_scaled = best_model.forecast(steps=len(test_data))
                
                # Generate future forecasts
                # Use the entire scaled dataset to make future predictions
                full_model = ARIMA(scaled_prices, order=best_order)
                full_fitted_model = full_model.fit()
                future_forecast_scaled = full_fitted_model.forecast(steps=future_days)
                
                # Transform back to original scale
                test_forecast = scaler.inverse_transform(test_forecast_scaled.reshape(-1, 1)).flatten()
                future_forecast = scaler.inverse_transform(future_forecast_scaled.reshape(-1, 1)).flatten()
                
                # Calculate metrics for test period
                test_actual = df['Close'][train_size:].values
                rmse = np.sqrt(mean_squared_error(test_actual, test_forecast))
                mae = mean_absolute_error(test_actual, test_forecast)
                r2 = r2_score(test_actual, test_forecast)
                accuracy = 100 * (1 - (rmse / np.mean(test_actual)))
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
                    st.metric("R² Score", f"{r2:.3f}", help="Coefficient of Determination")
                with col2:
                    st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                    st.metric("Forecast Accuracy", f"{accuracy:.2f}%", help="Percentage of accuracy in forecast")
                
                # Create future dates
                if has_datetime_index:
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                               periods=future_days, freq='D')
                else:
                    future_dates = list(range(len(df), len(df) + future_days))
                
                # Forecast visualization
                st.subheader("📈 Forecast vs Actual")
                
                # Create comprehensive plot
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Full Time Series with Forecasts', 
                                  'Test Period Validation (Zoomed)'),
                    vertical_spacing=0.2,
                    row_heights=[1, 0.5]
                )
                
                # Full time series plot
                fig.add_trace(go.Scatter(
                    x=train_dates, 
                    y=df['Close'][:train_size], 
                    mode='lines', 
                    name='Training Data',
                    line=dict(color='blue', width=1.5),
                    hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_actual, 
                    mode='lines', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_forecast, 
                    mode='lines', 
                    name='Test Forecast',
                    line=dict(color='green', width=2),
                    hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_forecast, 
                    mode='lines', 
                    name='Future Forecast',
                    line=dict(color='orange', width=2),
                    hovertemplate='Date: %{x}<br>Future Forecast: %{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                # Zoomed test period
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_actual, 
                    mode='lines+markers', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_forecast, 
                    mode='lines+markers', 
                    name='Test Forecast',
                    line=dict(color='green', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_forecast, 
                    mode='lines+markers', 
                    name='Future Forecast',
                    line=dict(color='orange', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Future Forecast: %{y:.2f}<extra></extra>'
                ), row=2, col=1)
                
                fig.update_layout(
                    title=f'{st.session_state.stock_name} - ARIMA{best_order} Forecast Results',
                    height=1200,
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
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Price", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals plot
                residuals = test_actual - test_forecast
                fg = px.bar(
                    x=test_dates, 
                    y=residuals, 
                    title='Residuals of ARIMA Forecast (Test Period)',
                    labels={'x': 'Date', 'y': 'Residuals'},
                )
                fg.update_layout(height=400)
                st.plotly_chart(fg, use_container_width=True)
                
                # Additional metrics
                st.subheader("📈 Historical Prediction Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
                    st.metric("Std Residual", f"{np.std(residuals):.4f}")
                
                with col2:
                    st.metric("Min Error", f"{np.min(residuals):.2f}")
                    st.metric("Max Error", f"{np.max(residuals):.2f}")
                
                # Forecast tables
                
                # Test period results
                test_forecast_df = pd.DataFrame({
                    'Date': test_dates,
                    'Actual': test_actual,
                    'Forecast': test_forecast,
                    'Residual': residuals,
                    'Absolute_Error': np.abs(residuals),
                    'Percentage_Error': (residuals / test_actual) * 100
                })
                
                # Format the test dataframe
                test_forecast_df['Actual'] = test_forecast_df['Actual'].round(2)
                test_forecast_df['Forecast'] = test_forecast_df['Forecast'].round(2)
                test_forecast_df['Residual'] = test_forecast_df['Residual'].round(4)
                test_forecast_df['Absolute_Error'] = test_forecast_df['Absolute_Error'].round(4)
                test_forecast_df['Percentage_Error'] = test_forecast_df['Percentage_Error'].round(2)
                
                
                with st.expander("📊 Validation Details", expanded=True):
                    st.dataframe(test_forecast_df, use_container_width=True)
                
                
                # Summary of future predictions
                st.subheader("🔮 Future Price Predictions")

                current_price = df['Close'].iloc[-1]
                price_change = future_forecast[-1] - current_price
                price_change_pct = (price_change / current_price) * 100
                
                future_forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_forecast,
                    'Price Change': future_forecast - current_price,
                    'Price Change (%)': ((future_forecast - current_price) / current_price) * 100
                })
                
                future_fig = px.line(
                future_forecast_df,
                x='Date', 
                y='Predicted Price', 
                title='Future Price Predictions',
                markers=True,
                color_discrete_sequence=['orange'],
                line_shape='linear',
                hover_data={'Date': True, 'Predicted Price': ':.2f'},
                labels={'x': 'Date', 'y': 'Predicted Price '},
                )
                future_fig.update_layout(height=400)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(future_fig, use_container_width=True)

                # Future predictions table
                with col2:
                    st.dataframe(future_forecast_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during forecasting: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error during ARIMA model training: {str(e)}")
            st.write("💡 Try adjusting the parameters or check your data for any issues.")
            st.write("Common issues: insufficient data, too many parameters, or non-numeric data.")

with st.expander("ℹ️ About ARIMA Models & Scaling"):
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)** models are used for time series forecasting.
    
    **Components:**
    - **AR (p)**: Autoregressive component - uses past values to predict future values
    - **I (d)**: Integrated component - degree of differencing to make data stationary
    - **MA (q)**: Moving average component - uses past forecast errors in prediction
    """)