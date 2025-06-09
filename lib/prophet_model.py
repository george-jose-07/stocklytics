import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import warnings

# Suppress Prophet warnings
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('prophet').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

st.title("🔮 Prophet Stock Price Forecasting and Prediction")
st.write("Facebook Prophet for time series forecasting with automatic hyperparameter tuning")

# Check if data is available
if st.session_state.df is None:
    st.warning("⚠️ Please upload data first from the 'Data Upload & Visualization' page")
    st.stop()

df = st.session_state.df.copy()
stock_name = st.session_state.stock_name

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
    st.metric("Stock Name", stock_name)


st.write("Date Range:", date_range_str)
st.write("---")

df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Model validation settings
st.subheader("Train-Test Split Configuration")
train_ratio = st.slider("Training Data Ratio (for validation)", 60, 90, 80, 1,
                                help="Used for model validation before making future predictions")
train_ratio = train_ratio / 100.0  # Convert to fraction
train_size = int(len(df) * train_ratio)
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test_actual = test['y'].values if len(test) > 0 else None

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Data Size", len(train))
with col2:
    st.metric("Testing Data Size", len(test))

# Visualize train-test split
fig_split = go.Figure()
fig_split.add_trace(go.Scatter(
    x=train['ds'], 
    y=train['y'], 
    mode='lines', 
    name='Training Data',
    line=dict(color='blue')
))
fig_split.add_trace(go.Scatter(
    x=test['ds'], 
    y=test['y'], 
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

st.header("🤖 LSTM Model Training & Forecasting")
st.header("Prophet Parameters")

# Prediction settings
st.subheader("Prediction Settings")
prediction_days = st.slider("Days to Predict", 1, 30, 7, 1, 
                                   help="Number of days into the future to predict")

# Hyperparameter tuning options
st.subheader("Hyperparameter Tuning")
enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=True, 
                                   help="Automatically find best parameters (takes longer)")

if enable_tuning:
    # Quick vs Comprehensive tuning
    tuning_mode = st.selectbox("Tuning Mode", 
                                      ["Quick", "Comprehensive"], 
                                      help="Quick: fewer parameter combinations, Comprehensive: more thorough search")
else:
    # Manual parameter selection
    st.subheader("Manual Parameters")
    col1, col2 = st.columns(2)
    with col1:
        changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001,
                                               help="Flexibility of trend changes")
    with col2:
        seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 20.0, 10.0, 0.01,
                                               help="Strength of seasonality")
    seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
    col1, col2 = st.columns(2)
    with col1:
        weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
    with col2:
        yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
    holidays_prior_scale = st.slider("Holidays Prior Scale", 0.01, 20.0, 10.0, 0.01)

# Advanced options
with st.expander("Advanced Options"):
    freq = st.selectbox("Frequency", ["B", "D"], index=0, 
                       help="B: Business days, D: Daily")
    uncertainty_samples = st.slider("Uncertainty Samples", 0, 1000, 1000,
                                   help="Number of samples for uncertainty intervals")

if st.button("🚀 Run Prophet Forecast & Predict Next Week", type="primary"):
    with st.spinner("Running Prophet forecasting and predicting future prices... This may take several minutes."):
        try:
                                
            best_rmse = float('inf')
            best_params = None
            all_results = []
            
            if enable_tuning and test_actual is not None and len(test_actual) > 0:
                # Define parameter grids based on tuning mode
                if tuning_mode == "Quick":
                    param_grid = {
                        'changepoint_prior_scale': [0.005, 0.05, 0.1],
                        'seasonality_prior_scale': [1.0, 10.0],
                        'seasonality_mode': ['additive', 'multiplicative'],
                        'daily_seasonality': [False],
                        'weekly_seasonality': [True, False],
                        'yearly_seasonality': [True],
                        'holidays_prior_scale': [1.0, 10.0],
                    }
                else:  # Comprehensive
                    param_grid = {
                        'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                        'seasonality_prior_scale': [0.01, 1.0, 10.0, 15.0],
                        'seasonality_mode': ['additive', 'multiplicative'],
                        'daily_seasonality': [False],
                        'weekly_seasonality': [True, False],
                        'yearly_seasonality': [True, False],
                        'holidays_prior_scale': [0.01, 1.0, 10.0],
                    }
                
                # Calculate total combinations
                total_combinations = np.prod([len(v) for v in param_grid.values()])

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Hyperparameter search
                current_combination = 0
                for params in product(*param_grid.values()):
                    current_params = dict(zip(param_grid.keys(), params))
                    current_combination += 1
                    
                    # Update progress
                    progress = current_combination / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing combination {current_combination}/{total_combinations}")
                    
                    try:
                        # Initialize and fit Prophet model on training data
                        model = Prophet(**current_params, uncertainty_samples=uncertainty_samples)
                        model.fit(train)
                        
                        # Make predictions for validation period
                        future = model.make_future_dataframe(periods=len(test), freq=freq)
                        forecast = model.predict(future)
                        
                        # Extract test predictions
                        current_prophet_forecast = forecast.iloc[-len(test):]['yhat'].values
                        
                        # Calculate RMSE
                        rmse = np.sqrt(mean_squared_error(test_actual, current_prophet_forecast))
                        all_results.append({'params': current_params.copy(), 'rmse': rmse})
                        
                        # Update best parameters
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = current_params.copy()
                            
                    except Exception as e:
                        all_results.append({'params': current_params.copy(), 'rmse': float('inf'), 'error': str(e)})
                
                progress_bar.progress(1.0)
                status_text.text("✅ Hyperparameter tuning completed!")
                
            else:
                # Use manual parameters or skip tuning if no validation data
                best_params = {
                    'changepoint_prior_scale': changepoint_prior_scale if not enable_tuning else 0.05,
                    'seasonality_prior_scale': seasonality_prior_scale if not enable_tuning else 10.0,
                    'seasonality_mode': seasonality_mode if not enable_tuning else 'additive',
                    'daily_seasonality': False,
                    'weekly_seasonality': weekly_seasonality if not enable_tuning else True,
                    'yearly_seasonality': yearly_seasonality if not enable_tuning else True,
                    'holidays_prior_scale': holidays_prior_scale if not enable_tuning else 10.0,
                }
                
                if enable_tuning:
                    st.info("⚠️ Insufficient validation data for tuning. Using default parameters.")
            
            # Train final model on ALL available data for future predictions
            if best_params:
                final_prophet_model = Prophet(**best_params, uncertainty_samples=uncertainty_samples)
                final_prophet_model.fit(df)  # Use all data for final model
                
                # Create future dataframe for predictions
                future_df = final_prophet_model.make_future_dataframe(periods=prediction_days, freq=freq)
                final_forecast_df = final_prophet_model.predict(future_df)
                
                # Extract historical and future predictions
                historical_forecast = final_forecast_df.iloc[:-prediction_days]['yhat'].values
                future_predictions = final_forecast_df.iloc[-prediction_days:]['yhat'].values
                future_lower = final_forecast_df.iloc[-prediction_days:]['yhat_lower'].values
                future_upper = final_forecast_df.iloc[-prediction_days:]['yhat_upper'].values
                future_dates = final_forecast_df.iloc[-prediction_days:]['ds'].values
                
                # Validation metrics (if validation data exists)
                rmse = None
                if test_actual is not None and len(test_actual) > 0:
                    validation_forecast = final_forecast_df.iloc[train_size:len(df)]['yhat'].values
                    if len(validation_forecast) == len(test_actual):
                        rmse = np.sqrt(mean_squared_error(test_actual, validation_forecast))
                        mae = mean_absolute_error(test_actual, validation_forecast)


                # Show best parameters if tuning was enabled
                if enable_tuning and best_params:
                    st.subheader("🎯 Best Parameters Found" if rmse else "🎯 Parameters Used")
                    params_df = pd.DataFrame.from_dict(best_params, orient='index', columns=['Value'])
                    params_df.index.name = 'Parameter'
                    with st.expander("📊 Best Parameters"):
                        st.dataframe(params_df, use_container_width=True)

                # Metrics
                # Key prediction insights
                last_price = df['y'].iloc[-1]
                last_prediction = future_predictions[-1]
                total_change = last_prediction - last_price
                total_change_pct = (total_change / last_price) * 100
                col1, col2, col3 = st.columns(3)
                with col1:
                    if rmse:
                        st.metric("RMSE", f"{rmse:.2f}")
                    else:
                        st.metric("Model Status", "Trained")
                with col2:
                    st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                with col3:
                    st.metric(f"Predicted Price (Day {prediction_days})", f"${last_prediction:.2f}", 
                             f"${total_change_pct:+.2f}%")
                
                st.subheader("📈 Forecast vs Actual")
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Full Time Series with Forecast', 'Forecast and Predictions (Zoomed)'),
                    vertical_spacing=0.2,
                    row_heights=[1, 0.5]
                )
                # Full time series plot
                fig.add_trace(go.Scatter(
                    x=train['ds'], 
                    y=train['y'], 
                    mode='lines', 
                    name='Training Data',
                    line=dict(color='blue', width=1.5),
                    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test['ds'], 
                    y=test['y'], 
                    mode='lines', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=test['ds'], 
                    y=validation_forecast, 
                    mode='lines', 
                    name='Prophet Forecast',
                    line=dict(color='green', width=2),
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_predictions.flatten(), 
                    mode='lines+markers', 
                    name='Future Predictions',
                    line=dict(color='orange', width=2, dash='dash'),
                    marker=dict(size=4),
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Zoomed forecast period
                fig.add_trace(go.Scatter(
                    x=test['ds'], 
                    y=test['y'], 
                    mode='lines+markers', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
                ), row=2, col=1)

                fig.add_trace(go.Scatter(
                    x=test['ds'], 
                    y=validation_forecast, 
                    mode='lines+markers', 
                    name='LSTM Forecast',
                    line=dict(color='green', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=2, col=1)

                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_predictions.flatten(), 
                    mode='lines+markers', 
                    name='Future Predictions',
                    line=dict(color='orange', width=2, dash='dash'),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=2, col=1)

                fig.update_layout(
                    title=f'{st.session_state.stock_name} - LSTM Forecast Results',
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
            
                # Future Predictions Table
                current_price = df['y'].iloc[-1]
                st.subheader("📅 Future Price Predictions")
                future_df_display = pd.DataFrame({
                    'Date': pd.to_datetime(future_dates),
                    'Predicted Price': future_predictions,
                    'Price Change': future_predictions - current_price,
                    'Change %': ((future_predictions - current_price) / current_price) * 100,
                })
                
                # Format the dataframe for better display
                future_df_display['Predicted Price'] = future_df_display['Predicted Price'].round(2)
                future_df_display['Date'] = future_df_display['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(future_df_display, use_container_width=True)

                # Model validation details (if available)
                if rmse and test_actual is not None:
                        validation_forecast_full = final_forecast_df.iloc[train_size:len(df)]['yhat'].values
                        
                        validation_details_df = pd.DataFrame({
                            'Date': test['ds'],
                            'Actual': test_actual,
                            'Predicted': validation_forecast_full,
                            'Residuals': test_actual - validation_forecast_full,
                            'Residuals': test_actual - validation_forecast_full,
                            'Abs Residuals': np.abs(test_actual - validation_forecast_full),
                            'Percentage_Error': ((test_actual - validation_forecast_full) / test_actual) * 100,
                        })
                          # Validation statistics
                        st.subheader("📈 Historical Prediction Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Residual", f"{np.mean(validation_details_df['Residuals']):.2f}")
                            st.metric("Std Residual", f"{np.std(validation_details_df['Residuals']):.2f}")
                        with col2:
                            st.metric("Min Error", f"{np.min(validation_details_df['Residuals']):.2f}")
                            st.metric("Max Error", f"{np.max(validation_details_df['Residuals']):.2f}")

                        st.dataframe(validation_details_df, use_container_width=True)
                
            else:
                st.error("❌ No valid parameters found. Please try different parameter ranges.")
                
        except Exception as e:
            st.error(f"❌ Error during Prophet forecasting: {str(e)}")
            st.write("**Possible solutions:**")
            st.write("- Check if your data has a proper date column")
            st.write("- Ensure you have enough data points (at least 30 days recommended)")
            st.write("- Try reducing the parameter search space")
            st.write("- Disable hyperparameter tuning and use manual parameters")
            st.write("- Check if your date column is properly formatted")

# Information about Prophet
with st.expander("ℹ️ About Prophet Forecasting & Future Predictions"):
    st.write("""
    **Facebook Prophet for Future Price Prediction**
    
    This enhanced version of Prophet focuses on predicting future stock prices rather than just evaluating historical performance.
    
    **Key Features:**
    - **Future Predictions**: Predicts actual future prices (next 1-30 days)
    - **Uncertainty Quantification**: Provides confidence intervals for predictions
    - **Automatic Hyperparameter Tuning**: Finds optimal parameters using historical validation
    - **Component Analysis**: Shows trend, seasonality, and other factors
    - **Risk Assessment**: Analyzes prediction uncertainty and confidence
    
    **How It Works:**
    1. **Validation Phase**: Uses historical data to find best parameters (if tuning enabled)
    2. **Training Phase**: Trains final model on ALL available data
    3. **Prediction Phase**: Generates future predictions with confidence intervals
    
    **Prediction Reliability:**
    - **1-7 days**: Generally most reliable
    - **1-2 weeks**: Good reliability for trending stocks
    - **2-4 weeks**: Moderate reliability, higher uncertainty
    - **Beyond 1 month**: Lower reliability, use with caution
    
    **Key Metrics:**
    - **Predicted Price**: Most likely future price
    - **Confidence Interval**: Range of possible prices (typically 80% confidence)
    - **Relative Uncertainty**: Confidence range as % of predicted price
    - **Coverage %**: How often actual prices fall within predicted ranges (from validation)
    
    **Best Practices:**
    - Use at least 3-6 months of historical data
    - Enable hyperparameter tuning for better accuracy
    - Consider external factors not captured by the model
    - Use predictions as guidance, not absolute truth
    - Monitor prediction accuracy over time
    
    **Risk Disclaimer:** These predictions are statistical estimates based on historical patterns. 
    Actual stock prices are influenced by many factors not captured in historical price data alone. 
    Always consider multiple sources and risk factors when making investment decisions.
    """)