import streamlit as st

# Configure page
st.set_page_config(
    page_title="Stock Forecasting App",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'stock_name' not in st.session_state:
    st.session_state.stock_name = "Stock"

# Define the pages
main_page = st.Page("lib/main.py", title="Data Upload & Visualization", icon="📊")
arima_page = st.Page("lib/arima.py", title="ARIMA Forecasting", icon="📈")
sarima_page = st.Page("lib/sarima.py", title="SARIMA Forecasting", icon="🔄")
lstm_page = st.Page("lib/lstm.py", title="LSTM Forecasting", icon="🧠")
rnn_page = st.Page("lib/rnn.py", title="RNN Forecasting", icon="🔗")
cnn_page = st.Page("lib/cnn.py", title="CNN Forecasting", icon="🖼️")
gru_page = st.Page("lib/gru.py", title="GRU Forecasting", icon="🌐")
lstm_gru_page = st.Page("lib/lstm_gru.py", title="LSTM-GRU Forecasting", icon="⚡")
prophet_page = st.Page("lib/prophet_model.py", title="Prophet Forecasting", icon="🔮")
lstm_cnn_rnn_page = st.Page("lib/lstm_cnn_rnn.py", title="LSTM-CNN-RNN Forecasting", icon="🤖")

# Set up navigation
pg = st.navigation([main_page, arima_page, sarima_page, lstm_page, rnn_page, cnn_page, gru_page, prophet_page, lstm_gru_page, lstm_cnn_rnn_page])

# Run the selected page
pg.run() 