import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from itertools import product
import logging
import warnings
from datetime import datetime, timedelta

# Suppress Prophet warnings
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('prophet').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

st.title("🔮 Prophet Stock Price Forecasting - Next Week Prediction")
st.write("Facebook Prophet for time series forecasting with automatic hyperparameter tuning")

# Check if data is available
if st.session_state.df is None:
    st.warning("⚠️ Please upload data first from the 'Data Upload & Visualization' page")
    st.stop()

df = st.session_state.df.copy()
stock_name = st.session_state.stock_name

# Sidebar for parameters
st.sidebar.header("Prophet Parameters")

# Prediction settings
st.sidebar.subheader("Prediction Settings")
prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 7, 1, 
                                   help="Number of days into the future to predict")

# Model validation settings
train_ratio = st.sidebar.slider("Training Data Ratio (for validation)", 0.6, 0.9, 0.8, 0.05,
                                help="Used for model validation before making future predictions")

# Hyperparameter tuning options
st.sidebar.subheader("Hyperparameter Tuning")
enable_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=True, 
                                   help="Automatically find best parameters (takes longer)")

if enable_tuning:
    # Quick vs Comprehensive tuning
    tuning_mode = st.sidebar.selectbox("Tuning Mode", 
                                      ["Quick", "Comprehensive"], 
                                      help="Quick: fewer parameter combinations, Comprehensive: more thorough search")
else:
    # Manual parameter selection
    st.sidebar.subheader("Manual Parameters")
    changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001,
                                               help="Flexibility of trend changes")
    seasonality_prior_scale = st.sidebar.slider("Seasonality Prior Scale", 0.01, 20.0, 10.0, 0.01,
                                               help="Strength of seasonality")
    seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"])
    weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
    yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
    holidays_prior_scale = st.sidebar.slider("Holidays Prior Scale", 0.01, 20.0, 10.0, 0.01)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    freq = st.selectbox("Frequency", ["B", "D"], index=0, 
                       help="B: Business days, D: Daily")
    uncertainty_samples = st.slider("Uncertainty Samples", 0, 1000, 1000,
                                   help="Number of samples for uncertainty intervals")

if st.button("🚀 Run Prophet Forecast & Predict Next Week", type="primary"):
    with st.spinner("Running Prophet forecasting and predicting future prices... This may take several minutes."):
        try:
            # Prepare data
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # For model validation (find best parameters)
            train_size = int(len(df) * train_ratio)
            prophet_train_df = prophet_df.iloc[:train_size]
            prophet_test_df = prophet_df.iloc[train_size:]
            test_actual = prophet_test_df['y'].values if len(prophet_test_df) > 0 else None
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Data Points", len(prophet_df))
            with col2:
                st.metric("Training Points", len(prophet_train_df))
            with col3:
                st.metric("Validation Points", len(prophet_test_df) if test_actual is not None else 0)
            with col4:
                st.metric("Future Predictions", prediction_days)
                
            try:
                st.info(f"📅 Data Range: {prophet_df['ds'].min().strftime('%Y-%m-%d')} to {prophet_df['ds'].max().strftime('%Y-%m-%d')}")
            except AttributeError as e:
                st.error(f"Date formatting error: {e}. Ensure 'ds' column is in datetime format.")
                st.stop()
                                
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
                st.info(f"🔍 Testing {total_combinations} parameter combinations for validation...")
                
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
                        model.fit(prophet_train_df)
                        
                        # Make predictions for validation period
                        future = model.make_future_dataframe(periods=len(prophet_test_df), freq=freq)
                        forecast = model.predict(future)
                        
                        # Extract test predictions
                        current_prophet_forecast = forecast.iloc[-len(prophet_test_df):]['yhat'].values
                        
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
                st.info("🔧 Training final model on all available data...")
                final_prophet_model = Prophet(**best_params, uncertainty_samples=uncertainty_samples)
                final_prophet_model.fit(prophet_df)  # Use all data for final model
                
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
                validation_rmse = None
                if test_actual is not None and len(test_actual) > 0:
                    validation_forecast = final_forecast_df.iloc[train_size:len(prophet_df)]['yhat'].values
                    if len(validation_forecast) == len(test_actual):
                        validation_rmse = np.sqrt(mean_squared_error(test_actual, validation_forecast))
                
                # Display results
                st.success(f"✅ Prophet model trained and future predictions generated!")
                
                # Show best parameters if tuning was enabled
                if enable_tuning and best_params:
                    st.subheader("🎯 Best Parameters Found" if validation_rmse else "🎯 Parameters Used")
                    params_df = pd.DataFrame.from_dict(best_params, orient='index', columns=['Value'])
                    params_df.index.name = 'Parameter'
                    st.dataframe(params_df, use_container_width=True)
                    
                    # Show tuning results summary
                    if len(all_results) > 0:
                        with st.expander("📊 Tuning Results Summary"):
                            results_df = pd.DataFrame(all_results)
                            valid_results = results_df[results_df['rmse'] != float('inf')]
                            
                            if len(valid_results) > 0:
                                st.write(f"**Successful combinations:** {len(valid_results)}/{len(all_results)}")
                                st.write(f"**Best RMSE:** {valid_results['rmse'].min():.2f}")
                                st.write(f"**Worst RMSE:** {valid_results['rmse'].max():.2f}")
                                st.write(f"**Average RMSE:** {valid_results['rmse'].mean():.2f}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if validation_rmse:
                        st.metric("Validation RMSE", f"{validation_rmse:.2f}")
                    else:
                        st.metric("Model Status", "Trained")
                with col2:
                    st.metric("Prediction Days", prediction_days)
                with col3:
                    st.metric("Model Type", "Prophet" + (" (Tuned)" if enable_tuning and len(all_results) > 0 else " (Default)"))
                
                # Future Predictions Table
                st.subheader("📊 Next Week Price Predictions")
                future_df_display = pd.DataFrame({
                    'Date': pd.to_datetime(future_dates),
                    'Predicted Price': future_predictions,
                    'Lower Bound': future_lower,
                    'Upper Bound': future_upper,
                    'Confidence Range': future_upper - future_lower
                })
                
                # Format the dataframe for better display
                future_df_display['Predicted Price'] = future_df_display['Predicted Price'].round(2)
                future_df_display['Lower Bound'] = future_df_display['Lower Bound'].round(2)
                future_df_display['Upper Bound'] = future_df_display['Upper Bound'].round(2)
                future_df_display['Confidence Range'] = future_df_display['Confidence Range'].round(2)
                future_df_display['Date'] = future_df_display['Date'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(future_df_display, use_container_width=True)
                
                # Key prediction insights
                last_price = prophet_df['y'].iloc[-1]
                first_prediction = future_predictions[0]
                last_prediction = future_predictions[-1]
                total_change = last_prediction - last_price
                total_change_pct = (total_change / last_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${last_price:.2f}")
                with col2:
                    st.metric("First Prediction", f"${first_prediction:.2f}", 
                             f"{first_prediction - last_price:+.2f}")
                with col3:
                    st.metric("Final Prediction", f"${last_prediction:.2f}")
                with col4:
                    st.metric("Total Change", f"{total_change_pct:+.1f}%", 
                             f"${total_change:+.2f}")
                
                # Main forecast plot
                fig, ax = plt.subplots(figsize=(16, 10))
                
                # Plot historical data
                ax.plot(prophet_df['ds'], prophet_df['y'], 
                       label='Historical Prices', color='blue', alpha=0.8, linewidth=2)
                
                # Plot historical model fit
                ax.plot(prophet_df['ds'], historical_forecast, 
                       label='Model Fit', color='orange', alpha=0.7, linewidth=1.5, linestyle='-.')
                
                # Plot future predictions
                ax.plot(pd.to_datetime(future_dates), future_predictions, 
                       label='Future Predictions', color='red', linewidth=3, marker='o', markersize=4)
                
                # Plot confidence intervals for future
                ax.fill_between(pd.to_datetime(future_dates), future_lower, future_upper,
                               color='red', alpha=0.2, label='Prediction Confidence Interval')
                
                # Add vertical line to separate historical and future
                ax.axvline(x=prophet_df['ds'].iloc[-1], color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                          label='Today')
                
                ax.set_title(f'{stock_name} - Prophet Forecast: Next {prediction_days} Days', 
                           fontsize=18, fontweight='bold', pad=20)
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Price ($)', fontsize=14)
                ax.legend(loc='upper left', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Improve date formatting on x-axis
                fig.autofmt_xdate()
                
                st.pyplot(fig)
                
                # Component plots
                st.subheader("📈 Forecast Components Analysis")
                
                # Get components
                components_fig = final_prophet_model.plot_components(final_forecast_df)
                st.pyplot(components_fig)
                
                # Model validation details (if available)
                if validation_rmse and test_actual is not None:
                    with st.expander("📊 Model Validation Details"):
                        validation_forecast_full = final_forecast_df.iloc[train_size:len(prophet_df)]['yhat'].values
                        validation_lower = final_forecast_df.iloc[train_size:len(prophet_df)]['yhat_lower'].values
                        validation_upper = final_forecast_df.iloc[train_size:len(prophet_df)]['yhat_upper'].values
                        
                        validation_details_df = pd.DataFrame({
                            'Date': prophet_test_df['ds'],
                            'Actual': test_actual,
                            'Predicted': validation_forecast_full,
                            'Lower Bound': validation_lower,
                            'Upper Bound': validation_upper,
                            'Error': test_actual - validation_forecast_full,
                            'Abs Error': np.abs(test_actual - validation_forecast_full),
                            'Within Bounds': (test_actual >= validation_lower) & (test_actual <= validation_upper)
                        })
                        
                        st.dataframe(validation_details_df, use_container_width=True)
                        
                        # Validation statistics
                        st.write("**Validation Statistics:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Absolute Error", f"{np.mean(validation_details_df['Abs Error']):.2f}")
                        with col2:
                            st.metric("Mean Error", f"{np.mean(validation_details_df['Error']):.2f}")
                        with col3:
                            st.metric("Max Absolute Error", f"{np.max(validation_details_df['Abs Error']):.2f}")
                        with col4:
                            coverage = np.mean(validation_details_df['Within Bounds']) * 100
                            st.metric("Coverage %", f"{coverage:.1f}%")
                
                # Prediction insights and analysis
                with st.expander("🔍 Prediction Analysis & Insights"):
                    st.write("**Prediction Summary:**")
                    st.write(f"- **Direction:** {'📈 Upward' if total_change > 0 else '📉 Downward' if total_change < 0 else '➡️ Sideways'}")
                    st.write(f"- **Magnitude:** {abs(total_change_pct):.1f}% change expected over {prediction_days} days")
                    st.write(f"- **Daily Average Change:** {total_change_pct/prediction_days:.2f}% per day")
                    
                    # Volatility analysis
                    avg_confidence_range = np.mean(future_df_display['Confidence Range'])
                    avg_prediction = np.mean(future_predictions)
                    relative_uncertainty = (avg_confidence_range / avg_prediction) * 100
                    
                    st.write(f"\n**Uncertainty Analysis:**")
                    st.write(f"- **Average Confidence Range:** ${avg_confidence_range:.2f}")
                    st.write(f"- **Relative Uncertainty:** {relative_uncertainty:.1f}% of predicted price")
                    st.write(f"- **Risk Level:** {'High' if relative_uncertainty > 10 else 'Medium' if relative_uncertainty > 5 else 'Low'}")
                    
                    st.write("\n**Model Parameters Used:**")
                    for param, value in best_params.items():
                        st.write(f"- **{param}**: {value}")
                
                # Download predictions
                csv_data = future_df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_data,
                    file_name=f"{stock_name}_prophet_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
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