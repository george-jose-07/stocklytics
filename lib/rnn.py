import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

st.title("📊 RNN Stock Price Forecasting and Prediction")
st.write("Recurrent Neural Network (RNN) for time series prediction")

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


# for parameters
st.subheader("Train-Test Split Configuration")
train_ratio = st.slider("Training Data Ratio",60,90, 80, 1)
train_ratio = train_ratio / 100.0  # Convert to decimal for calculations

# Prepare data
train_size = int(len(df) * train_ratio)
train = df.iloc[:train_size]['Close']
test = df.iloc[train_size:]['Close']

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

# Visualize train-test split
fig_split = go.Figure()
fig_split.add_trace(go.Scatter(
    x=train_dates, 
    y=train, 
    mode='lines', 
    name='Training Data',
    line=dict(color='blue')
))
fig_split.add_trace(go.Scatter(
    x=test_dates, 
    y=test, 
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
st.header("🤖 RNN Model Training & Forecasting")
st.subheader("RNN Parameters")

col1, col2 = st.columns(2)
with col1:
    lookback = st.slider("Lookback Window", 10, 60, 30, 5)
with col2:
    rnn_units = st.slider("RNN Units", 25, 100, 50, 5)

batch_size = st.selectbox("Batch Size", [2, 4, 8, 16, 32, 64])

epochs = st.slider("Training Epochs", 0, 100, 30, 5)

future_days = st.slider("Days to Predict into Future", 1, 30, 14, 1)  # Add option for future prediction days

if st.button("🚀 Run RNN Forecast", type="primary"):
    with st.spinner("Training RNN model... This may take a while."):
        try:
            # Scale data
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(df[['Close']])
            
            # Create sequences
            X_train, y_train = [], []
            train_scaled = scaled_data[:train_size]
            
            for i in range(lookback, len(train_scaled)):
                X_train.append(train_scaled[i-lookback:i, 0])
                y_train.append(train_scaled[i, 0])
            
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Build model with RNN layers instead of LSTM
            model = Sequential()
            model.add(SimpleRNN(units=rnn_units, return_sequences=True, input_shape=(X_train.shape[1],1)))
            model.add(Dropout(0.2))
            model.add(SimpleRNN(units=rnn_units))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train the model
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
                
            progress_bar.progress(1.0)  # Complete the progress bar
            status_text.text('Training completed!')
            
            # Predict on test set
            inputs = scaled_data[train_size-lookback:]
            X_test = []
            
            for i in range(lookback, len(inputs)):
                X_test.append(inputs[i-lookback:i, 0])
            
            X_test = np.array(X_test)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            predictions = model.predict(X_test, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            
            # Calculate RMSE
            mse = mean_squared_error(test.values, predictions)
            rnn_rmse = np.sqrt(mean_squared_error(test.values, predictions))
            mae = mean_absolute_error(test, predictions)
            r2 = r2_score(test, predictions)
            accuracy = (1 - (mae / np.mean(test))) * 100

            # Use the last 'lookback' days from the entire dataset for future prediction
            last_sequence = scaled_data[-lookback:]
            future_predictions = []
            
            for _ in range(future_days):
                # Reshape for prediction
                X_future = last_sequence.reshape((1, lookback, 1))
                
                # Predict next value
                next_pred = model.predict(X_future, verbose=0)
                future_predictions.append(next_pred[0, 0])
                
                # Update sequence for next prediction (sliding window)
                last_sequence = np.append(last_sequence[1:], next_pred[0, 0]).reshape(-1, 1)
            
            # Convert future predictions back to original scale
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)
            
            # Create future dates
            last_date = df.index[-1]
            future_dates = []
            for i in range(1, future_days + 1):
                if hasattr(last_date, 'date'):
                    future_date = last_date + timedelta(days=i)
                else:
                    future_date = pd.to_datetime(last_date) + timedelta(days=i)
                future_dates.append(future_date)
            
            # Display results
            st.success(f"✅ RNN model trained successfully!")
            st.write("---")
            st.header("📈 Forecasting Results")
            col1, col2  = st.columns(2)
            with col1:
                st.metric("RMSE", f"{rnn_rmse:.2f}", help="Root Mean Square Error")
                st.metric("R²", f"{r2:.2f}", help="R-squared")
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
            # Full time series plot
            fig.add_trace(go.Scatter(
                    x=train_dates, 
                    y=train, 
                    mode='lines', 
                    name='Training Data',
                    line=dict(color='blue', width=1.5),
                    hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
                
            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test, 
                    mode='lines', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
                
            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=predictions.flatten(), 
                    mode='lines', 
                    name='RNN Forecast',
                    line=dict(color='green', width=2),
                    hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y= future_predictions.flatten(), 
                    mode='lines', 
                    name='Future Predictions',
                    line=dict(color='orange', width=2),
                    hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

            # Zoomed forecast period
            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=test, 
                    mode='lines+markers', 
                    name='Actual (Test)',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                    x=test_dates, 
                    y=predictions.flatten(), 
                    mode='lines+markers', 
                    name='RNN Forecast',
                    line=dict(color='green', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y= future_predictions.flatten(), 
                    mode='lines+markers', 
                    name='Future Predictions',
                    line=dict(color='orange', width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
            ), row=2, col=1)

            fig.update_layout(
                    title=f'{st.session_state.stock_name} - RNN Forecast Results',
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
            fig.update_yaxes(title_text="Price ", row=1, col=1)
            fig.update_yaxes(title_text="Price ", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)
            

            residuals = test.values - predictions.flatten()
            fg = px.bar(
                df,
                x=test_dates, 
                y=residuals, 
                title='Residuals of RNN Forecast',
                labels={'x': 'Date', 'y': 'Residuals'},
                )
            fg.update_layout(height=400)
            st.plotly_chart(fg, use_container_width=True)

            # Prediction statistics
            st.subheader("📈 Historical Prediction Statistics")
            pred_df = pd.DataFrame({
                    'Date': test.index,
                    'Actual': test.values,
                    'Predicted': predictions.flatten(),
                    'Residuals': residuals,
                    'Abs Error': np.abs(residuals),
                    'Percentage_Error': (residuals / test.values) * 100
            })

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Residual", f"{np.mean(pred_df['Residuals']):.2f}")
                st.metric("Std Residual", f"{np.std(pred_df['Residuals']):.2f}")
            with col2:
                st.metric("Min Error", f"{np.min(pred_df['Residuals']):.2f}")
                st.metric("Max Error", f"{np.max(pred_df['Residuals']):.2f}")
                
            with st.expander("📊 Validation Details", expanded=True):
                st.dataframe(pred_df)

            # Future predictions table
            current_price = df['Close'].iloc[-1]
            future_price = future_predictions[-1, 0]
            price_change = future_price - current_price
            change_pct = (price_change / current_price) * 100
            st.subheader("📅 Future Price Predictions")

            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_predictions.flatten(),
                'Price Change': future_predictions.flatten() - current_price,
                'Change %': ((future_predictions.flatten() - current_price) / current_price) * 100
            })

            
            future_fig = px.line(
                future_df,
                x='Date', 
                y='Predicted Price', 
                title='Future Price Predictions',
                markers=True,
                color_discrete_sequence=['orange'],
                line_shape='linear',
                hover_data={'Date': True, 'Predicted Price': ':.2f'},
                labels={'x': 'Date', 'y': 'Predicted Price'},
                )
            future_fig.update_layout(height=400)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(future_fig, use_container_width=True)
            with col2:
                st.dataframe(future_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error during RNN training: {str(e)}")

# Information about RNN
with st.expander("ℹ️ About RNN Forecasting"):
    st.write("""
    **Recurrent Neural Network (RNN)** is a type of neural network designed for sequential data and time series forecasting.
    
    **Key Features:**
    - Processes sequential data step by step
    - Maintains information through hidden states
    - Suitable for time series prediction tasks
    - Simpler architecture compared to LSTM
    
    **Parameters:**
    - **Lookback Window**: Number of previous time steps to use for prediction
    - **RNN Units**: Number of neurons in each RNN layer
    - **Epochs**: Number of training iterations
    - **Batch Size**: Number of samples processed together
        """)