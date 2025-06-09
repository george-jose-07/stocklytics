import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Stock Data Upload & Visualization")

# File uploader
file = st.file_uploader("Choose a CSV file", type=["csv", "txt"])
if file is not None:
    try:
        df = pd.read_csv(file)
            
        # Try to identify and parse date column
        date_column = None
        potential_date_columns = ['Date', 'date', 'DATE', 'Datetime', 'datetime', 'Time', 'time']
        for col in potential_date_columns:
            if col in df.columns:
                date_column = col
                break
            
            # If date column found, parse it and set as index
        if date_column:
            try:
                df[date_column] = pd.to_datetime(df[date_column]).dt.date
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.set_index(date_column)
                df = df.sort_index()  # Sort by date
            except Exception as e:
                st.warning(f"Could not parse date column '{date_column}': {e}")
                st.info("Using numeric indices for plotting.")
        else:
            st.warning("No date column found. Using numeric indices for plotting.")
            st.info("For better visualization, include a 'Date' column in your CSV.")
            
        # Store in session state
        st.session_state.df = df
            
            # Display basic info about the dataset
        st.write("### Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
            
        # Show column names
        st.write("**Columns:**", ", ".join(df.columns.tolist()))
            
        # Show date range if available
        if date_column and df.index.dtype.kind == 'M':  # Check if index is datetime
            st.write(f"**Date Range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            
        # Display first few rows
        st.write("### First 10 rows of the dataset:")
        st.dataframe(df.head(10))
            
        # Check if required columns exist
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
            
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Please ensure your CSV has at least a 'Close' column for ARIMA analysis.")
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
                
            # Plot for closing price 
            st.subheader("Stock Closing Price Over Time")
                
            # Create a copy for plotting to avoid index issues
            plot_df = df.reset_index()
            x_column = date_column if date_column and date_column in plot_df.columns else plot_df.columns[0]
                
            fig_close = px.line(
                plot_df, 
                x=x_column, 
                y='Close', 
                title=f'{st.session_state.stock_name} - Closing Price Over Time',
                labels={x_column: 'Date' if date_column else 'Time Period', 'Close': 'Closing Price'}
            )
            fig_close.update_layout(height=400)
            st.plotly_chart(fig_close, use_container_width=True)
                
            # Plot for OHLC if columns exist
            ohlc_columns = ['Open', 'High', 'Low', 'Close']
            available_ohlc = [col for col in ohlc_columns if col in df.columns]
                
            if len(available_ohlc) > 1:
                st.subheader("OHLC Prices Over Time")
                fig_ohlc = px.line(
                    plot_df, 
                    x=x_column, 
                    y=available_ohlc, 
                    title=f'{st.session_state.stock_name} - OHLC Prices Over Time',
                    labels={x_column: 'Date' if date_column else 'Time Period', 'value': 'Price'}
                )
                fig_ohlc.update_layout(height=400)
                st.plotly_chart(fig_ohlc, use_container_width=True)
                
            # Volume plot if available
            if 'Volume' in df.columns:
                st.subheader("Trading Volume Over Time")
                fig_volume = px.bar(
                    plot_df, 
                    x=x_column, 
                    y='Volume', 
                    title=f'{st.session_state.stock_name} - Trading Volume Over Time',
                    labels={x_column: 'Date' if date_column else 'Time Period', 'Volume': 'Trading Volume'}
                )
                fig_volume.update_layout(height=400)
                st.plotly_chart(fig_volume, use_container_width=True)
                
            # Basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(df.describe())
                
            # Navigation hint
            st.write("---")
                
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
else:
    st.info("👆 Please upload a CSV file to proceed with analysis.")
        

