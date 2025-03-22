import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import io
import re
import base64
from pypdf import PdfReader

# Page config
st.set_page_config(page_title="DAJANIII", page_icon="📈", layout="wide")

# Custom header
st.markdown('''
    <style>
    .dajaniii-header {
        background-color: #2E7D32;
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
''', unsafe_allow_html=True)

st.markdown('<div class="dajaniii-header"><h1>DAJANIII</h1><h3>Stock Trading Assistant</h3></div>', unsafe_allow_html=True)

@st.cache_data
def load_sample_portfolio():
    return pd.DataFrame({
        'Stock': ['AAPL', 'MSFT', 'GOOGL'],
        'Shares': [10, 5, 3],
        'Purchase Price': [140.0, 270.0, 2400.0],
        'Term': ['Long', 'Short', 'Long']
    })

@st.cache_data
def get_current_prices(symbols):
    prices = {}
    for sym in symbols:
        try:
            data = yf.Ticker(sym).history(period="1d")
            prices[sym] = data['Close'].iloc[-1] if not data.empty else 0
        except:
            prices[sym] = 0
    return prices
def calculate_metrics(df, current_prices):
    df = df.copy()
    df['Current Price'] = df['Stock'].map(current_prices)
    df['Current Value'] = df['Shares'] * df['Current Price']
    df['Purchase Value'] = df['Shares'] * df['Purchase Price']
    df['Unrealized Gain'] = df['Current Value'] - df['Purchase Value']
    df['Gain %'] = (df['Unrealized Gain'] / df['Purchase Value']) * 100
    df['Holding Type'] = df['Term'].apply(lambda x: 'Long-term' if x.lower() == 'long' else 'Short-term')
    df['Tax Rate'] = df['Holding Type'].apply(lambda x: 0.15 if x == 'Long-term' else 0.35)
    df['Estimated Tax'] = df.apply(lambda x: x['Unrealized Gain'] * x['Tax Rate'] if x['Unrealized Gain'] > 0 else 0, axis=1)
    return df

# Sidebar
st.sidebar.header("Upload Portfolio")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.button("Use Sample Portfolio")

# Load portfolio
portfolio_df = None
if uploaded_file:
    try:
        portfolio_df = pd.read_csv(uploaded_file)
        st.success("Portfolio uploaded!")
    except:
        st.error("Failed to read uploaded CSV.")
elif use_sample:
    portfolio_df = load_sample_portfolio()
    st.success("Sample portfolio loaded.")
if portfolio_df is not None:
    st.subheader("📊 Portfolio Overview")

    symbols = portfolio_df['Stock'].tolist()
    current_prices = get_current_prices(symbols)
    portfolio_metrics = calculate_metrics(portfolio_df, current_prices)

    col1, col2, col3 = st.columns(3)
    total_value = portfolio_metrics['Current Value'].sum()
    total_gain = portfolio_metrics['Unrealized Gain'].sum()
    total_tax = portfolio_metrics['Estimated Tax'].sum()

    with col1:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Unrealized Gain", f"${total_gain:,.2f}")
    with col3:
        st.metric("Est. Tax", f"${total_tax:,.2f}")

    st.dataframe(portfolio_metrics[[
        'Stock', 'Shares', 'Purchase Price', 'Current Price',
        'Unrealized Gain', 'Gain %', 'Holding Type', 'Estimated Tax'
    ]].style.format({
        'Purchase Price': '${:.2f}',
        'Current Price': '${:.2f}',
        'Unrealized Gain': '${:.2f}',
        'Gain %': '{:.2f}%',
        'Estimated Tax': '${:.2f}'
    }), use_container_width=True)
else:
    st.info("Upload a CSV portfolio or click 'Use Sample Portfolio' to begin.")
# Optional: Sector Breakdown (basic placeholder logic)
st.subheader("📈 Sector Allocation (Placeholder)")

# For a real app, you would fetch sector info via an API or mapping
sector_data = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Communication Services'
}

if portfolio_df is not None:
    sector_df = portfolio_df.copy()
    sector_df['Sector'] = sector_df['Stock'].map(sector_data).fillna('Other')
    sector_alloc = sector_df.groupby('Sector')['Current Value'].sum().reset_index()
    
    fig = px.pie(sector_alloc, values='Current Value', names='Sector', title='Portfolio by Sector')
    st.plotly_chart(fig, use_container_width=True)

# Optional Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "DAJANIII Stock Assistant &copy; 2024 – Powered by Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
