import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
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

# Prepare data
train_ratio = st.slider("Training Data Ratio (for validation)", 60, 90, 80, 1)
train_ratio = train_ratio / 100.0  # Convert to fraction
train_size = int(len(df) * train_ratio)
train_data = df.iloc[:train_size]['Close']
test_data = df.iloc[train_size:]['Close']

# Get date indices for plotting
if has_datetime_index:
    train_dates = df.index[:train_size]
    test_dates = df.index[train_size:]
    full_dates = df.index
else:
    train_dates = list(range(len(train_data)))
    test_dates = list(range(len(train_data), len(df)))
    full_dates = list(range(len(df)))

# Prepare data for Prophet format (Prophet requires 'ds' and 'y' columns)
prophet_df = pd.DataFrame()
prophet_df['ds'] = df.index
prophet_df['y'] = df['Close'].values

train = prophet_df.iloc[:train_size]
test = prophet_df.iloc[train_size:]
test_actual = test_data.values if len(test_data) > 0 else None

# Model validation settings
st.subheader("Train-Test Split Configuration")

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Data Size", len(train_data))
with col2:
    st.metric("Testing Data Size", len(test_data))

# Visualize train-test split using train_dates, test_dates
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

st.header("🔮 Prophet Model Training & Forecasting")
st.header("Prophet Parameters")


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
                model = Prophet(**best_params, uncertainty_samples=uncertainty_samples)
                model.fit(train)

                # Create future dataframe for predictions
                future_df = model.make_future_dataframe(periods=len(test), freq=freq)
                forecast = model.predict(future_df)
                
                # # Extract historical and future predictions
                # historical_forecast = final_forecast_df.iloc[:-prediction_days]['yhat'].values
                # future_predictions = final_forecast_df.iloc[-prediction_days:]['yhat'].values
                # future_dates = final_forecast_df.iloc[-prediction_days:]['ds'].values
                
                # Validation metrics (if validation data exists)

                validation_forecast = forecast.iloc[-len(test):]['yhat'].values
                rmse = np.sqrt(mean_squared_error(test_actual, validation_forecast))
                mae = mean_absolute_error(test_actual, validation_forecast)
                r2 = r2_score(test_actual, validation_forecast)
                accuracy = (1-(mae/ np.mean(test_actual))) * 100
                #accuracy = accuracy_score(np.round(test_actual), np.round(validation_forecast))
                
                # Show best parameters if tuning was enabled
                if enable_tuning and best_params:
                    st.subheader("🎯 Best Parameters Found" if rmse else "🎯 Parameters Used")
                    params_df = pd.DataFrame.from_dict(best_params, orient='index', columns=['Value'])
                    params_df.index.name = 'Parameter'
                    with st.expander("📊 Best Parameters"):
                        st.dataframe(params_df, use_container_width=True)

                # # Key prediction insights using Close prices
                # last_price = df['Close'].iloc[-1]
                # last_prediction = future_predictions[-1]
                # total_change = last_prediction - last_price
                # total_change_pct = (total_change / last_price) * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{rmse:.2f}")
                    st.metric("R²", f"{r2:.2f}")

                with col2:
                    st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
                    st.metric("Accuracy", f"{accuracy:.2f}%", help="Prediction Accuracy")

                st.subheader("📈 Forecast vs Actual")
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Full Time Series with Forecast', 'Forecast and Predictions (Zoomed)'),
                    vertical_spacing=0.2,
                    row_heights=[1, 0.5]
                )
                
                # Full time series plot using train_dates, test_dates
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
                    
                    # Add validation forecast if available
                fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=validation_forecast, 
                    mode='lines', 
                    name='Prophet Forecast',
                    line=dict(color='green', width=2),
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
                
                # Future predictions
                # fig.add_trace(go.Scatter(
                #     x=future_dates, 
                #     y=future_predictions.flatten(), 
                #     mode='lines+markers', 
                #     name='Future Predictions',
                #     line=dict(color='orange', width=2, dash='dash'),
                #     marker=dict(size=4),
                #     hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                # ), row=1, col=1)

                # Zoomed forecast period (only if test data exists)
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
                    y=validation_forecast, 
                    mode='lines+markers', 
                    name='Prophet Forecast',
                    line=dict(color='green', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                ), row=2, col=1)

                    # fig.add_trace(go.Scatter(
                    #     x=future_dates, 
                    #     y=future_predictions.flatten(), 
                    #     mode='lines+markers', 
                    #     name='Future Predictions',
                    #     line=dict(color='orange', width=2, dash='dash'),
                    #     marker=dict(size=4),
                    #     showlegend=False,
                    #     hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
                    # ), row=2, col=1)

                fig.update_layout(
                    title=f'{st.session_state.stock_name} - Prophet Forecast Results',
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
            
                # Future Predictions Table using Close prices
                # current_price = df['Close'].iloc[-1]
                # st.subheader("📅 Future Price Predictions")
                # future_df_display = pd.DataFrame({
                #     'Date': pd.to_datetime(future_dates),
                #     'Predicted Price': future_predictions,
                #     'Price Change': future_predictions - current_price,
                #     'Change %': ((future_predictions - current_price) / current_price) * 100,
                # })
                
                # # Format the dataframe for better display
                # future_df_display['Predicted Price'] = future_df_display['Predicted Price'].round(2)
                # future_df_display['Price Change'] = future_df_display['Price Change'].round(2)
                # future_df_display['Change %'] = future_df_display['Change %'].round(2)
                # future_df_display['Date'] = future_df_display['Date'].dt.strftime('%Y-%m-%d')
                # st.dataframe(future_df_display, use_container_width=True)

                # Model validation details (if available)
                residuals = test_actual - validation_forecast
                fg = px.bar(
                df,
                x=test_dates, 
                y=residuals, 
                title='Residuals of Prophet Forecast',
                labels={'x': 'Date', 'y': 'Residuals'},
                )
                fg.update_layout(height=400)
                st.plotly_chart(fg, use_container_width=True)

                validation_details_df = pd.DataFrame({
                        'Date': test_dates,
                        'Actual': test_actual,
                        'Predicted': validation_forecast,
                        'Residuals': residuals,
                        'Abs Residuals': np.abs(residuals),
                        'Percentage_Error': (residuals / test_actual) * 100,
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

                st.subheader("📋 Detailed Forecast Results")
                st.dataframe(validation_details_df, use_container_width=True)
                
                
        except Exception as e:
            st.error(f"❌ Error during Prophet forecasting: {str(e)}")

# Information about Prophet
with st.expander("ℹ️ About Prophet Forecasting"):
    st.write("""
    **Facebook Prophet for Future Price Prediction**
    
    This enhanced version of Prophet focuses on predicting future stock prices rather than just evaluating historical performance.
    
    **Key Features:**
    - **Uncertainty Quantification**: Provides confidence intervals for predictions
    - **Automatic Hyperparameter Tuning**: Finds optimal parameters using historical validation
    - **Component Analysis**: Shows trend, seasonality, and other factors
    - **Risk Assessment**: Analyzes prediction uncertainty and confidence
    
    **How It Works:**
    1. **Validation Phase**: Uses historical data to find best parameters (if tuning enabled)
    2. **Training Phase**: Trains final model on ALL available data
    3. **Prediction Phase**: Generates future predictions with confidence intervals
    """)