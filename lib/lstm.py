import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.title("📊 LSTM Stock Price Forecasting with Next Week Prediction")
st.write("Long Short-Term Memory (LSTM) neural network for time series prediction")

# Check if data is available
if st.session_state.df is None:
    st.warning("⚠️ Please upload data first from the 'Data Upload & Visualization' page")
    st.stop()

df = st.session_state.df.copy()
stock_name = st.session_state.stock_name

# Sidebar for parameters
st.sidebar.header("LSTM Parameters")
train_ratio = st.sidebar.slider("Training Data Ratio", 0.6, 0.9, 0.8, 0.05)
lookback = st.sidebar.slider("Lookback Window", 10, 60, 30, 5)
lstm_units = st.sidebar.slider("LSTM Units", 25, 100, 50, 25)
epochs = st.sidebar.slider("Training Epochs", 1, 50, 5, 1)
batch_size = st.sidebar.selectbox("Batch Size", [1, 2, 4, 8, 16, 32])

# Add option for future prediction days
future_days = st.sidebar.slider("Days to Predict into Future", 1, 14, 7, 1)

if st.button("🚀 Run LSTM Forecast", type="primary"):
    with st.spinner("Training LSTM model... This may take a while."):
        try:
            # Prepare data
            train_size = int(len(df) * train_ratio)
            train = df.iloc[:train_size]['Close']
            test = df.iloc[train_size:]['Close']
            
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
            
            # Build model
            model = Sequential()
            model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1],1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=lstm_units))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train the model
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            progress_bar.progress(1.0)
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
            lstm_rmse = np.sqrt(mean_squared_error(test.values, predictions))
            
            # === NEW: Predict future prices ===
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
            st.success(f"✅ LSTM model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{lstm_rmse:.2f}")
            with col2:
                st.metric("Training Data Points", len(train))
            with col3:
                current_price = df['Close'].iloc[-1]
                future_price = future_predictions[-1, 0]
                price_change = future_price - current_price
                change_pct = (price_change / current_price) * 100
                st.metric(
                    f"Predicted Price (Day {future_days})", 
                    f"${future_price:.2f}",
                    f"{change_pct:+.2f}%"
                )
            
            # Plot results with future predictions
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Historical data
            ax.plot(train.index, train.values, label='Training Data', color='blue', alpha=0.7)
            ax.plot(test.index, test.values, label='Actual Prices', color='red', linewidth=2)
            ax.plot(test.index, predictions.flatten(), label='LSTM Forecast', color='green', linewidth=2, linestyle='--')
            
            # Future predictions
            ax.plot(future_dates, future_predictions.flatten(), label=f'Future Forecast ({future_days} days)', 
                   color='orange', linewidth=3, linestyle='-', marker='o', markersize=4)
            
            # Add vertical line to separate historical from future
            ax.axvline(x=df.index[-1], color='gray', linestyle=':', alpha=0.7, label='Present')
            
            ax.set_title(f'{stock_name} - LSTM Forecasting with Future Predictions', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Separate future predictions graph
            st.subheader("🔮 Next Week Prediction Focus")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Show last 30 days + future predictions
            recent_data = df.tail(30)
            ax2.plot(recent_data.index, recent_data['Close'], label='Recent Actual Prices', 
                    color='blue', linewidth=2, marker='o', markersize=3)
            ax2.plot(future_dates, future_predictions.flatten(), label=f'Future Predictions', 
                    color='red', linewidth=3, marker='o', markersize=5)
            
            # Connect last actual price to first prediction
            ax2.plot([recent_data.index[-1], future_dates[0]], 
                    [recent_data['Close'].iloc[-1], future_predictions[0, 0]], 
                    color='gray', linestyle='--', alpha=0.7)
            
            ax2.axvline(x=df.index[-1], color='gray', linestyle=':', alpha=0.7, label='Present')
            ax2.set_title(f'{stock_name} - Future Price Predictions', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Price ($)', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Future predictions table
            st.subheader("📅 Future Price Predictions")
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_predictions.flatten(),
                'Price Change': future_predictions.flatten() - current_price,
                'Change %': ((future_predictions.flatten() - current_price) / current_price) * 100
            })
            
            # Format the dataframe for better display
            future_df['Predicted Price'] = future_df['Predicted Price'].apply(lambda x: f"${x:.2f}")
            future_df['Price Change'] = future_df['Price Change'].apply(lambda x: f"${x:+.2f}")
            future_df['Change %'] = future_df['Change %'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(future_df, use_container_width=True)
            
            # Training loss plot
            st.subheader("📊 Training Loss")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(history.history['loss'], color='blue', linewidth=2)
            ax3.set_title('Model Training Loss', fontsize=14)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            
            # Model summary
            with st.expander("🔍 Model Architecture"):
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
            
            # Prediction statistics
            with st.expander("📈 Historical Prediction Statistics"):
                pred_df = pd.DataFrame({
                    'Date': test.index,
                    'Actual': test.values,
                    'Predicted': predictions.flatten(),
                    'Error': test.values - predictions.flatten()
                })
                
                st.dataframe(pred_df.tail(10))
                
                st.write("**Error Statistics:**")
                st.write(f"- Mean Absolute Error: {np.mean(np.abs(pred_df['Error'])):.2f}")
                st.write(f"- Mean Error: {np.mean(pred_df['Error']):.2f}")
                st.write(f"- Max Error: {np.max(np.abs(pred_df['Error'])):.2f}")
            
            # Risk disclaimer
            st.warning("⚠️ **Investment Disclaimer**: These predictions are based on historical data and machine learning models. Stock prices are influenced by many factors not captured in historical price data alone. Please do not use these predictions as the sole basis for investment decisions.")
            
        except Exception as e:
            st.error(f"❌ Error during LSTM training: {str(e)}")
            st.write("Please check your data and parameters.")

# Information about LSTM
with st.expander("ℹ️ About LSTM Forecasting"):
    st.write("""
    **Long Short-Term Memory (LSTM)** is a type of recurrent neural network that's particularly effective for time series forecasting.
    
    **Key Features:**
    - Handles long-term dependencies in sequential data
    - Learns complex patterns and relationships
    - Can capture non-linear trends
    
    **Parameters:**
    - **Lookback Window**: Number of previous time steps to use for prediction
    - **LSTM Units**: Number of neurons in each LSTM layer
    - **Epochs**: Number of training iterations
    - **Batch Size**: Number of samples processed together
    - **Future Days**: Number of days to predict into the future
    
    **Future Prediction Method:**
    - Uses the last 'lookback' days to predict the next day
    - Applies a sliding window approach for multi-day predictions
    - Each prediction becomes part of the input for the next prediction
    
    **Note**: LSTM models may take longer to train compared to traditional methods like ARIMA. Future predictions become less reliable the further into the future they extend.
    """)

with st.expander("🎯 How Future Predictions Work"):
    st.write("""
    **Multi-Step Ahead Forecasting:**
    
    1. **Initial Input**: Use the last N days (lookback window) from your dataset
    2. **First Prediction**: Model predicts the next day's price
    3. **Sliding Window**: Remove the oldest day, add the predicted day
    4. **Repeat**: Use this new sequence to predict the day after
    5. **Continue**: Repeat this process for the desired number of future days
    
    **Important Considerations:**
    - Prediction accuracy typically decreases with longer forecast horizons
    - The model only uses price history, not external factors (news, earnings, etc.)
    - Results should be combined with fundamental analysis for investment decisions
    
    **Interpreting Results:**
    - Green line: Historical model performance on test data
    - Orange/Red line: Future predictions beyond available data
    - The further the prediction, the higher the uncertainty
    """)