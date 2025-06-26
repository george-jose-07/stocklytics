import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import warnings
warnings.filterwarnings('ignore')

st.title("🔄 SARIMA Model Forecasting")
st.write("SARIMA (Seasonal AutoRegressive Integrated Moving Average) model for time series forecasting.")


# Check if data is available
if st.session_state.df is None:
    st.warning("⚠️ Please upload data first from the 'Data Upload & Visualization' page")
    st.stop()

df = st.session_state.df.copy()
stock_name = st.session_state.stock_name

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
    st.metric("Stock Name", stock_name)


st.write("Date Range:", date_range_str)
st.write("---")

st.subheader("Train-Test Split Configuration")

train_ratio = st.slider("Training Data Ratio", 60, 90, 80,1)
train_ratio = train_ratio / 100.0

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale the Close prices
close_scaled = scaler.fit_transform(df[['Close']])
df_scaled = df.copy()
df_scaled['Close'] = close_scaled.flatten()

# Prepare scaled data
train_size = int(len(df_scaled) * train_ratio)
train = df_scaled.iloc[:train_size]['Close']
test = df_scaled.iloc[train_size:]['Close']

# Get original scale data for display
train_original = df.iloc[:train_size]['Close']
test_original = df.iloc[train_size:]['Close']

# Get date indices for plotting
if has_datetime_index:
    train_dates = df.index[:train_size]
    test_dates = df.index[train_size:]
    full_dates = df.index
else:
    train_dates = list(range(len(train)))
    test_dates = list(range(len(train), len(df)))
    full_dates = list(range(len(df)))

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Data Size", len(train))
with col2:
    st.metric("Testing Data Size", len(test))

# Visualize train-test split (using original scale for display)
fig_split = go.Figure()
fig_split.add_trace(go.Scatter(
    x=train_dates, 
    y=train_original, 
    mode='lines', 
    name='Training Data',
    line=dict(color='blue')
))
fig_split.add_trace(go.Scatter(
    x=test_dates, 
    y=test_original, 
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


st.header("🤖 SARIMA Model Training & Forecasting")
future_days = st.slider("Days to Predict into Future", 1, 30, 14, 1)
auto_search = st.radio(
    "Parameter Selection",
    ["Manual Selection", "Grid Search"],
    help="Choose between manual parameter selection or automatic grid search"
)

if auto_search == "Manual Selection":
    st.subheader("ARIMA Parameters (p,d,q)")
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
    with col2:
        d = st.number_input("Integration order (d)", min_value=0, max_value=2, value=1)
    with col3:
        q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
    
    st.subheader("Seasonal Parameters (P,D,Q,s)")
    col1, col2, col3 = st.columns(3)
    with col1:
        P = st.number_input("Seasonal AR (P)", min_value=0, max_value=2, value=1)
    with col2:
        D = st.number_input("Seasonal Integration (D)", min_value=0, max_value=1, value=1)
    with col3:
        Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=2, value=1)

    s = st.slider("Seasonal Period (s)", 5, 50, 30, 1)
    
else:  # Grid Search
    st.subheader("Search Ranges")
    col1, col3 = st.columns(2)
    with col1:
        p_range = st.slider("AR order range (p)", 1, 5, (0, 2))
    with col3:
        q_range = st.slider("MA order range (q)", 1, 5, (0, 2))

    d_range = (0, 2)  # Fixed to 0 or 1 for simplicity

    col1, col3 = st.columns(2)
    with col1:
        P_range = st.slider("Seasonal AR range (P)", 1, 2, (0, 1))
    with col3:
        Q_range = st.slider("Seasonal MA range (Q)", 1, 2, (0, 1))

    D_range = (0, 1)  # Fixed to 0 or 1 for simplicity

    s = st.selectbox("Seasonal Period (s)", 
                     [5, 7, 10, 12, 14, 30, 50], 
                     index=3, 
                     help="Seasonal period (s) is the number of observations in one seasonal cycle. Common values are 5, 7, 10, 12, 14, 30, or 50.")

# Advanced options
with st.expander("Advanced Options"):
    information_criterion = st.selectbox("Information Criterion", ["aic", "bic"], index=0)
    include_trend = st.selectbox("Trend Component", ["c", "ct", "ctt", "n"], 
                                index=0, help="c=constant, ct=constant+trend, ctt=constant+linear+quadratic, n=none")

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
                
                forecast = fitted_model.forecast(steps=len(test))
                score = np.sqrt(mean_squared_error(test, forecast))
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
        try:
            # Parameter selection
            if auto_search == "Grid Search":
                st.subheader("🔍 Grid Search for Optimal Parameters")
                best_params, best_seasonal_params, best_score, search_results = grid_search_sarima(
                    train, p_range, d_range, q_range, P_range, D_range, Q_range, s, information_criterion
                )
                
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
            
            # Fit the final model on scaled data
            with st.spinner("Fitting final SARIMA model..."):
                model = SARIMAX(train,
                              order=(p, d, q),
                              seasonal_order=(P, D, Q, s),
                              trend=include_trend,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                
                fitted_model = model.fit(disp=False)
            
            # Generate forecasts on scaled data for test period
            forecast_scaled = fitted_model.forecast(steps=len(test))
            
            # For future predictions, fit model on entire dataset (train + test)
            with st.spinner("Fitting model on full dataset for future predictions..."):
                full_data_scaled = df_scaled['Close']  # Use entire scaled dataset
                future_model = SARIMAX(full_data_scaled,
                                     order=(p, d, q),
                                     seasonal_order=(P, D, Q, s),
                                     trend=include_trend,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)
                
                future_fitted_model = future_model.fit(disp=False)
                
            # Generate future forecast on scaled data using full dataset model
            future_forecast_scaled = future_fitted_model.forecast(steps=future_days)
            
            # Transform back to original scale
            forecast = scaler.inverse_transform(forecast_scaled.values.reshape(-1, 1)).flatten()
            future_forecast = scaler.inverse_transform(future_forecast_scaled.values.reshape(-1, 1)).flatten()
            
            if has_datetime_index:
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                               periods=future_days, freq='D')
            else:
                    future_dates = list(range(len(df), len(df) + future_days))

            # Calculate RMSE using original scale
            rmse = np.sqrt(mean_squared_error(test_original, forecast))
            mae = mean_absolute_error(test_original, forecast)
            r2 = r2_score(test_original, forecast)
            accuracy = (1 - (mae / np.mean(test_original))) * 100

            st.write("---")
            st.header("📈 Forecasting Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("R²", f"{r2:.2f}")
            with col2:
                st.metric("MAE", f"{mae:.2f}")
                st.metric("Forecast Accuracy", f"{accuracy:.2f}%")

            st.write(f"**Model:** SARIMA({p},{d},{q})({P},{D},{Q})[{s}]")
            # Main forecast plot
            st.subheader("📈 Forecast vs Actual")
            fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Full Time Series with Forecast', 'Forecast Period (Zoomed)'),
                    vertical_spacing=0.2,
                    row_heights=[1, 0.5]
            )
            # Full time series plot
            fig.add_trace(go.Scatter(
                    x=train_dates, 
                    y=train_original, 
                    mode='lines', 
                    name='Training Data',
                    line=dict(color='blue', width=1.5),
                    hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
                
            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_original, 
                    mode='lines', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
                
            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=forecast, 
                    mode='lines', 
                    name='SARIMA Forecast',
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

            # Zoomed forecast period
            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test_original, 
                    mode='lines+markers', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            ), row=2, col=1)
                
            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=forecast, 
                    mode='lines+markers', 
                    name='ARIMA Forecast',
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
                    title=f'{st.session_state.stock_name} - SARIMA({p},{d},{q})({P},{D},{Q})[{s}] Forecast Results',
                    height=1000,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
            )
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ", row=1, col=1)
            fig.update_yaxes(title_text="Price ", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            residuals = test_original.values - forecast
            # Residuals plot
            fg = px.bar(
                df,
                x=test_dates, 
                y=residuals, 
                title='Residuals of SARIMA Forecast',
                labels={'x': 'Date', 'y': 'Residuals'},
            )
            fg.update_layout(height=400)
            st.plotly_chart(fg, use_container_width=True)

            # Forecast statistics
            forecast_df = pd.DataFrame({
                'Date': test_original.index,
                'Actual': test_original.values,
                'Forecast': forecast,
                'Residuals': residuals,
                'Abs Error': np.abs(residuals),
                'Percentage_Error': (residuals / test_original.values) * 100
            })
            # Error statistics
            st.subheader("📈 Historical Prediction Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Residual", f"{np.mean(forecast_df['Residuals']):.2f}")
                st.metric("Std Residual", f"{np.std(forecast_df['Residuals']):.2f}")
            with col2:
                st.metric("Min Error", f"{np.min(forecast_df['Residuals']):.2f}")
                st.metric("Max Error", f"{np.max(forecast_df['Residuals']):.2f}")
                
            # Forecast table
            with st.expander("📊 Validation Details", expanded=True):
                st.dataframe(forecast_df)

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
            st.error(f"❌ Error during SARIMA modeling: {str(e)}")

# Information about SARIMA
with st.expander("ℹ️ About SARIMA Forecasting"):
    st.write("""
    **Seasonal ARIMA (SARIMA)** extends ARIMA modeling to handle seasonal patterns in time series data.
    
    **Model Structure:** SARIMA(p,d,q)(P,D,Q)[s]
    - **(p,d,q)**: Non-seasonal parameters (AR, Integration, MA)
    - **(P,D,Q)**: Seasonal parameters
    - **[s]**: Seasonal period
    
    **Parameters Guide:**
    - **Seasonal Period (s)**: Number of observations in one seasonal cycle
    - **Grid Search**: Automatically finds best parameters within specified ranges
    - **Manual Selection**: Allows precise control over model parameters
    - **Trend Component**: 
      - 'c': constant
      - 'ct': constant + linear trend
      - 'ctt': constant + linear + quadratic trend
      - 'n': no trend
    """)