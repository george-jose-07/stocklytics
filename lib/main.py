import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

st.title("Stock Data Upload & Visualization")

# Define required columns
REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# Add tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload CSV", "📊 Fetch from Yahoo Finance"])

with tab2:
    st.header("Fetch Stock Data from Yahoo Finance")

    col1, col2 = st.columns(2)

    with col1:
        stock_symbol = st.text_input(
            "Stock Symbol (e.g., AAPL, MSFT, GOOGL)", 
            value="AAPL",
            key="yf_symbol"
        )
    
    with col2:
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "10 Years": "10y",
            "Max": "max"
        }
        selected_period = st.selectbox(
            "Select Time Period",
            options=list(period_options.keys()),
            index=3,  # Default to 1 Year
            key="yf_period"
        )
    
    fetch_button = st.button("Fetch Data", type="primary", key="fetch_yf_data")
    
    if fetch_button and stock_symbol:
        try:
            with st.spinner(f"Fetching data for {stock_symbol.upper()}..."):
                # Fetch data using yfinance
                ticker = yf.Ticker(stock_symbol.upper())
                df = ticker.history(period=period_options[selected_period])
                
                if df.empty:
                    st.error(f"No data found for symbol '{stock_symbol.upper()}'. Please check the symbol and try again.")
                else:
                    # Reset index to make Date a column
                    df = df.reset_index()
                    
                    # Keep only required columns (yfinance provides all of them)
                    df = df[['Date'] + REQUIRED_COLUMNS]
                    
                    # Set Date as index and sort
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                    df = df.sort_index()
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.stock_name = stock_symbol.upper()
                    
                    st.success(f"Successfully fetched {len(df)} rows of data for {stock_symbol.upper()}")
                    
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.info("Please check your internet connection and ensure the stock symbol is valid.")

with tab1:
    st.header("Upload CSV File")
    
    # File uploader
    file = st.file_uploader("Choose a CSV file", type=["csv", "txt"])
    fetch_button2 = st.button("Fetch Data", type="primary")
    if file is not None and fetch_button2:
        try:
            df = pd.read_csv(file)
            # Try to identify and parse date column
            # date_column = None
            df = df.reset_index()
            df = df[['Date'] + REQUIRED_COLUMNS]
            # potential_date_columns = ['Date', 'date', 'DATE', 'Datetime', 'datetime', 'Time', 'time']
            # for col in potential_date_columns:
            #     if col in df.columns:
            #         date_column = col
            #         break
                
            # If date column found, parse it and set as index
            
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.sort_index()  # Sort by date
            st.session_state.df = df 
            st.session_state.stock_name = "Stock"  # Default stock name
            # else:
            #     st.warning("No date column found. Using numeric indices for plotting.")
            #     st.info("For better visualization, include a 'Date' column in your CSV.")
            
            # Filter to keep only required columns that exist in the CSV
            available_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            
            if not available_cols:
                st.error(f"None of the required columns found: {', '.join(REQUIRED_COLUMNS)}")
                st.info("Please ensure your CSV contains at least one of these columns: Open, High, Low, Close, Volume")
            else:
                # Keep only the required columns that are available
                df = df[available_cols]
                
                if missing_cols:
                    st.warning(f"Missing columns: {', '.join(missing_cols)}")
                
                st.success(f"Successfully loaded dataset with columns: {', '.join(available_cols)}")
                
                # Store in session state
                st.session_state.df = df
            
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    else:
        if 'df' not in st.session_state:
            st.info("👆 Please upload a CSV file to proceed with analysis.")

# Display data analysis if data is available
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    # Display basic info about the dataset
    st.write("### Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
        
        
    # Show date range if available
    if df.index.dtype.kind == 'M':  # Check if index is datetime
        st.write(f"**Date Range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
    # Display first few rows
    st.write("### First 10 rows of the dataset:")
    st.dataframe(df.head(10))
        
    # Check if Close column exists (minimum requirement)
    if 'Close' not in df.columns:
        st.error("Missing required 'Close' column for analysis.")
        st.info("Please ensure your dataset has at least a 'Close' column.")
    else:
        # Stock name input
        if 'stock_name' not in st.session_state:
            st.session_state.stock_name = "Stock"
            
        st.session_state.stock_name = st.text_input(
            "Enter Stock Name for Plot Title (Optional)", 
            value=st.session_state.stock_name
        )
            
        # Visualizations
        st.write("---")
        st.header("📈 Data Visualizations")
            
        # Plot for closing price (always available since we check for it)
        st.subheader("Stock Closing Price Over Time")
            
        # Create a copy for plotting to avoid index issues
        plot_df = df.reset_index()
        date_column = 'Date' if 'Date' in plot_df.columns else plot_df.columns[0]
        x_column = date_column if date_column and date_column in plot_df.columns else plot_df.columns[0]
            
        fig_close = px.line(
            plot_df, 
            x=x_column, 
            y='Close', 
            title=f'{st.session_state.stock_name} - Closing Price Over Time',
            labels={x_column: 'Date' if 'Date' in plot_df.columns else 'Time Period', 'Close': 'Closing Price'}
        )
        fig_close.update_layout(height=400)
        st.plotly_chart(fig_close, use_container_width=True)
            
        # Plot for OHLC only if we have the required columns
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        available_ohlc = [col for col in ohlc_columns if col in df.columns]
            
        if len(available_ohlc) >= 2:  # Need at least 2 columns for a meaningful OHLC chart
            st.subheader("OHLC Prices Over Time")
            fig_ohlc = px.line(
                plot_df, 
                x=x_column, 
                y=available_ohlc, 
                title=f'{st.session_state.stock_name} - OHLC Prices Over Time',
                labels={x_column: 'Date' if 'Date' in plot_df.columns else 'Time Period', 'value': 'Price'}
            )
            fig_ohlc.update_layout(height=400)
            st.plotly_chart(fig_ohlc, use_container_width=True)
        else:
            st.info("OHLC chart requires at least 2 price columns (Open, High, Low, Close)")
            
        # Volume plot only if Volume column exists
        if 'Volume' in df.columns:
            st.subheader("Trading Volume Over Time")
            fig_volume = px.bar(
                plot_df, 
                x=x_column, 
                y='Volume', 
                title=f'{st.session_state.stock_name} - Trading Volume Over Time',
                labels={x_column: 'Date' if 'Date' in plot_df.columns else 'Time Period', 'Volume': 'Trading Volume'}
            )
            fig_volume.update_layout(height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.info("Volume chart not available - 'Volume' column not found in dataset")

        x=df.copy()  # Use x for rolling mean calculations to avoid confusion with plot_df
        x['rolling mean 30'] = df['Close'].rolling(window=30).mean()
        x['rolling mean 7'] = df['Close'].rolling(window=7).mean()
        x['rolling mean 20'] = df['Close'].rolling(window=20).mean()
        x['rolling mean 10'] = df['Close'].rolling(window=10).mean()
        st.subheader("Rolling Mean of Closing Price")
        fig_rolling_mean = px.line(
            x.reset_index(),
            x='Date',
            y=['rolling mean 7', 'rolling mean 10', 'rolling mean 20', 'rolling mean 30'],
            title=f'{st.session_state.stock_name} - Rolling Mean of Closing Price',
            labels= {
                'Date': 'Date',
                'rolling mean 7': '7-Day Rolling Mean',
                'rolling mean 10': '10-Day Rolling Mean',
                'rolling mean 20': '20-Day Rolling Mean',
                'rolling mean 30': '30-Day Rolling Mean'}

        )
        st.plotly_chart(fig_rolling_mean, use_container_width=True)

        # corr = df[['Open','High','Low','Close','Volume']].corr()
        # fg_corr = px.imshow(
        #     corr,
        #     text_auto=True,
        #     title=f'{st.session_state.stock_name} - Correlation Matrix',
        #     labels=dict(x="Columns", y="Columns", color="Correlation Coefficient")
        # )
        # fg_corr.update_layout(height=400)
        # st.plotly_chart(fg_corr, use_container_width=True)

        # Basic statistics - only for available columns
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
        # Navigation hint
        st.write("---")
else:
    # Show message when no data is available
    st.write("---")
    st.info("📊 **No dataset loaded yet.**")
    st.write("Please either:")
    st.write("- Upload a CSV file using the '📁 Upload CSV' tab, or")
    st.write("- Fetch stock data using the '📊 Fetch from Yahoo Finance' tab")