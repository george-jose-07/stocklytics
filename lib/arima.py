import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

st.success("✅ Data loaded successfully for ARIMA analysis!")

# Check if we have datetime index
has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
if has_datetime_index:
    date_range_str = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
else:
    date_range_str = "No date information available"

# Display data info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Data Points", len(df))
with col2:
    st.metric("Stock Name", st.session_state.stock_name)
with col3:
    st.metric("Date Range", date_range_str)

# Show a snippet of the data
with st.expander("📋 View Data Sample"):
    st.dataframe(df[['Close']].head(10))

st.write("---")

# Data preparation section
st.header("🔧 Data Preparation")

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
    test_dates = list(range(len(train_data), len(df)))
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

# Stationarity Check
st.header("📊 Stationarity Analysis")

if STATSMODELS_AVAILABLE:
    def check_stationarity(timeseries):
        result = adfuller(timeseries.dropna())
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Is Stationary': result[1] <= 0.05
        }
    
    stationarity_result = check_stationarity(train_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ADF Statistic", f"{stationarity_result['ADF Statistic']:.4f}")
        st.metric("p-value", f"{stationarity_result['p-value']:.4f}")
    with col2:
        status = "✅ Stationary" if stationarity_result['Is Stationary'] else "❌ Non-Stationary"
        st.metric("Stationarity Status", status)
        
        # Show critical values
        crit_vals = stationarity_result['Critical Values']
        st.write("**Critical Values:**")
        for key, value in crit_vals.items():
            st.write(f"- {key}: {value:.4f}")
    
    if not stationarity_result['Is Stationary']:
        st.info("💡 Data appears to be non-stationary. ARIMA will apply differencing to make it stationary.")
    else:
        st.success("✅ Data is stationary and ready for ARIMA modeling!")

# ARIMA Model Section
st.header("🤖 ARIMA Model Training & Forecasting")

if not STATSMODELS_AVAILABLE:
    st.error("❌ statsmodels library is not installed. Please install it using: pip install statsmodels")
    st.stop()

# ARIMA Parameters
st.subheader("🔧 ARIMA Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    p = st.number_input("P (AR terms)", min_value=0, max_value=10, value=1, help="Autoregressive terms")
with col2:
    d = st.number_input("D (Differencing)", min_value=0, max_value=3, value=1, help="Degree of differencing")
with col3:
    q = st.number_input("Q (MA terms)", min_value=0, max_value=10, value=1, help="Moving average terms")

# Option to try multiple parameter combinations
auto_select = st.checkbox("🎯 Auto-select best parameters (tries multiple combinations)", value=True)
if auto_select:
    st.info("💡 Auto-selection will test different parameter combinations and choose the one with the lowest AIC score.")

# Train ARIMA Model
if st.button("🚀 Train ARIMA Model", type="primary"):
    with st.spinner("🔄 Training ARIMA Model... This may take a moment."):
        try:
            best_aic = float('inf')
            best_model = None
            best_order = None
            results_log = []
            
            if auto_select:
                # Try different parameter combinations
                st.info("🔍 Searching for best parameters...")
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
                    
                st.success(f"🎯 Best ARIMA parameters found: ARIMA{best_order} with AIC: {best_aic:.2f}")
                st.info(f"✅ Successfully fitted {successful_fits}/{total_combinations} parameter combinations")
                
                # Show top 5 best models
                if results_log:
                    results_df = pd.DataFrame(results_log)
                    # Filter out failed attempts and sort by AIC
                    successful_results = results_df[results_df['AIC'] != 'Failed'].copy()
                    if not successful_results.empty:
                        successful_results['AIC'] = pd.to_numeric(successful_results['AIC'])
                        successful_results = successful_results.sort_values('AIC').head(5)
                        
                        with st.expander("🏆 Top 5 Best Models"):
                            st.dataframe(successful_results, use_container_width=True)
                
            else:
                # Use user-specified parameters
                model = ARIMA(train_data, order=(p, d, q))
                best_model = model.fit()
                best_order = (p, d, q)
                best_aic = best_model.aic
            
            # Store model in session state
            st.session_state.arima_model = best_model
            st.session_state.arima_order = best_order
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            st.session_state.train_dates = train_dates
            st.session_state.test_dates = test_dates
            st.session_state.has_datetime_index = has_datetime_index
            
            st.success("✅ ARIMA Model trained successfully!")
            
            # Display model summary
            with st.expander("📊 View Model Summary"):
                st.text(str(best_model.summary()))
                
            # Model diagnostics
            with st.expander("🔍 Model Diagnostics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AIC (Akaike Information Criterion)", f"{best_aic:.2f}")
                    st.metric("BIC (Bayesian Information Criterion)", f"{best_model.bic:.2f}")
                with col2:
                    st.metric("Log Likelihood", f"{best_model.llf:.2f}")
                    st.metric("Parameters Count", len(best_model.params))
                
            st.info(f"🎯 Final ARIMA parameters: ARIMA{best_order}")
            
        except Exception as e:
            st.error(f"❌ Error during ARIMA model training: {str(e)}")
            st.info("💡 Try adjusting the parameters or check your data for any issues.")
            st.info("Common issues: insufficient data, too many parameters, or non-numeric data.")

# Forecasting section
if hasattr(st.session_state, 'arima_model') and st.session_state.arima_model is not None:
    st.write("---")
    st.header("📈 Forecasting Results")
    
    try:
        # Get stored data
        arima_model = st.session_state.arima_model
        test_data = st.session_state.test_data
        train_data = st.session_state.train_data
        train_dates = st.session_state.train_dates
        test_dates = st.session_state.test_dates
        has_datetime_index = st.session_state.has_datetime_index
        
        # Generate forecasts
        forecast_result = arima_model.forecast(steps=len(test_data))
        arima_forecast = forecast_result
        
        # Get confidence intervals if available
        forecast_ci = None
        try:
            forecast_ci = arima_model.get_forecast(steps=len(test_data)).conf_int()
        except:
            pass
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
        mae = mean_absolute_error(test_data, arima_forecast)
        mape = np.mean(np.abs((test_data - arima_forecast) / test_data)) * 100
        
        # Direction accuracy (for trend prediction)
        actual_direction = np.diff(test_data.values) > 0
        forecast_direction = np.diff(arima_forecast) > 0
        direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
        
        # Display metrics
        st.subheader("📊 Forecast Accuracy Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error")
        with col2:
            st.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
        with col4:
            st.metric("Direction Accuracy", f"{direction_accuracy:.1f}%", help="Percentage of correct trend direction predictions")
        
        # Forecast visualization
        st.subheader("📈 Forecast vs Actual")
        
        # Create comprehensive plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Full Time Series with Forecast', 'Forecast Period (Zoomed)'),
            vertical_spacing=0.12,
            row_heights=[0.65, 0.35]
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
            line=dict(color='green', dash='dash', width=2),
            hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        # Add confidence intervals if available
        if forecast_ci is not None:
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=forecast_ci.iloc[:, 1],
                mode='lines',
                line=dict(color='green', width=0),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=forecast_ci.iloc[:, 0],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                line=dict(color='green', width=0),
                name='Confidence Interval',
                hovertemplate='Date: %{x}<br>CI Lower: $%{y:.2f}<extra></extra>'
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
            line=dict(color='green', dash='dash', width=2),
            marker=dict(size=4),
            showlegend=False,
            hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
        ), row=2, col=1)
        
        fig.update_layout(
            title=f'{st.session_state.stock_name} - ARIMA{st.session_state.arima_order} Forecast Results',
            height=700,
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
        
        # Residuals analysis
        st.subheader("🔍 Residuals Analysis")
        residuals = test_data.values - arima_forecast
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals plot
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=test_dates, 
                y=residuals, 
                mode='lines+markers',
                name='Residuals',
                line=dict(color='purple'),
                marker=dict(size=4)
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Line")
            fig_residuals.update_layout(
                title='Residuals Over Time',
                xaxis_title='Date' if has_datetime_index else 'Time Period',
                yaxis_title='Residuals',
                height=400
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
            
        with col2:
            # Residuals histogram
            fig_hist = px.histogram(
                x=residuals, 
                nbins=20, 
                title='Residuals Distribution',
                labels={'x': 'Residuals', 'y': 'Frequency'}
            )
            fig_hist.add_vline(x=np.mean(residuals), line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: {np.mean(residuals):.3f}")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Additional metrics
        st.subheader("📊 Additional Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
            st.metric("Std Residual", f"{np.std(residuals):.4f}")
        
        with col2:
            st.metric("Min Error", f"{np.min(residuals):.2f}")
            st.metric("Max Error", f"{np.max(residuals):.2f}")
            
        with col3:
            # Calculate percentage of predictions within 5% of actual
            within_5_percent = np.mean(np.abs(residuals / test_data.values) <= 0.05) * 100
            st.metric("Within 5% Accuracy", f"{within_5_percent:.1f}%")
            
            # Calculate R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((test_data.values - np.mean(test_data.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            st.metric("R-squared", f"{r_squared:.4f}")
        
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
        
        # Download forecast results
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Results",
            data=csv,
            file_name=f"{st.session_state.stock_name}_arima{st.session_state.arima_order}_forecast.csv",
            mime="text/csv"
        )
        
        # Model interpretation
        st.subheader("💡 Model Interpretation")
        arima_order = st.session_state.arima_order
        interpretation_text = f"""
        **ARIMA{arima_order} Model Components:**
        - **AR({arima_order[0]})**: Uses {arima_order[0]} previous values to predict the next value
        - **I({arima_order[1]})**: Data was differenced {arima_order[1]} time(s) to make it stationary
        - **MA({arima_order[2]})**: Uses {arima_order[2]} previous forecast error(s) in the prediction
        
        **Performance Summary:**
        - RMSE of {rmse:.2f} indicates the average prediction error
        - MAPE of {mape:.2f}% shows the average percentage error
        - Direction accuracy of {direction_accuracy:.1f}% for trend prediction
        - R-squared of {r_squared:.4f} indicates the model explains {r_squared*100:.1f}% of the variance
        """
        st.markdown(interpretation_text)
        
    except Exception as e:
        st.error(f"❌ Error during forecasting: {str(e)}")
        st.info("💡 Try retraining the model or check your data quality.")

else:
    st.info("👆 Please train the ARIMA model first by clicking the 'Train ARIMA Model' button above.")
    
    # Show sample ARIMA explanation
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