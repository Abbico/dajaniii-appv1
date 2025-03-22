import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import time
import os
import re
import json
from openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import pypdf
from pypdf import PdfReader
import io
import base64

# Set page config
st.set_page_config(
    page_title="DAJANIII Stock Trading Assistant",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS
st.markdown('''
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .market-up {
        color: green;
        font-weight: bold;
    }
    .market-down {
        color: red;
        font-weight: bold;
    }
    .dajaniii-header {
        background-color: #2E7D32;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
''', unsafe_allow_html=True)

# Header
def display_header():
    st.markdown('<div class="dajaniii-header"><h1>DAJANIII</h1><h3>Advanced Stock Trading Assistant</h3></div>', unsafe_allow_html=True)
# Sample portfolio
@st.cache_data
def load_sample_portfolio():
    data = {
        'Stock': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'Shares': [10, 5, 2, 3, 8],
        'Purchase Price': [150.75, 280.50, 2750.25, 3300.10, 220.75],
        'Term': ['Long', 'Short', 'Long', 'Short', 'Long']
    }
    return pd.DataFrame(data)

# Schwab CSV parsing
@st.cache_data
def parse_schwab_portfolio(file_content):
    try:
        df = pd.read_csv(io.StringIO(file_content), skiprows=3)
        df.columns = [col.split('(')[0].strip() if '(' in col else col for col in df.columns]

        relevant_columns = {
            'Symbol': 'Stock',
            'Qty': 'Shares',
            'Price': 'Purchase Price',
            'Description': 'Description'
        }

        portfolio_df = pd.DataFrame()
        for schwab_col, std_col in relevant_columns.items():
            if schwab_col in df.columns:
                portfolio_df[std_col] = df[schwab_col]

        portfolio_df['Term'] = 'Long'
        if 'Shares' in portfolio_df.columns:
            portfolio_df['Shares'] = portfolio_df['Shares'].astype(str).str.replace(',', '').astype(float)
        if 'Purchase Price' in portfolio_df.columns:
            portfolio_df['Purchase Price'] = portfolio_df['Purchase Price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

        return portfolio_df
    except Exception as e:
        st.error(f"Error parsing Schwab portfolio: {str(e)}")
        return None
# Interactive Brokers PDF parsing
@st.cache_data
def parse_ib_portfolio(file_content):
    try:
        pdf_reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        stocks, shares, prices = [], [], []

        pattern = r"([A-Z]+)\s+([0-9,]+)\s+([0-9.]+)"
        matches = re.findall(pattern, text)

        for match in matches:
            if len(match) >= 3:
                stocks.append(match[0])
                shares.append(float(match[1].replace(',', '')))
                prices.append(float(match[2]))

        if stocks:
            return pd.DataFrame({
                'Stock': stocks,
                'Shares': shares,
                'Purchase Price': prices,
                'Term': ['Long'] * len(stocks)
            })
        else:
            return load_sample_portfolio()
    except Exception as e:
        st.error(f"Error parsing Interactive Brokers portfolio: {str(e)}")
        return None

# Get current stock prices
@st.cache_data(ttl=300)
def get_current_prices(symbols):
    if isinstance(symbols, str):
        symbols = [symbols]
    prices = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d")
            prices[symbol] = data['Close'].iloc[-1] if not data.empty else 0
        except:
            prices[symbol] = 0
    return prices

# Market index data
@st.cache_data(ttl=300)
def get_market_data():
    indexes = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'VIX': '^VIX'
    }
    market_data = {}
    for name, symbol in indexes.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                pct_change = (change / previous) * 100
                market_data[name] = {
                    'current': current,
                    'change': change,
                    'percent_change': pct_change
                }
        except:
            market_data[name] = {'current': 0, 'change': 0, 'percent_change': 0}
    return market_data
# Crypto data
@st.cache_data(ttl=300)
def get_crypto_data():
    crypto_symbols = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Solana': 'SOL-USD'
    }
    crypto_data = {}

    for name, symbol in crypto_symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="7d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                percent_change = (change / previous) * 100
                weekly_change = ((current / hist['Close'].iloc[0]) - 1) * 100
                crypto_data[name] = {
                    'current': current,
                    'change': change,
                    'percent_change': percent_change,
                    'week_change': weekly_change
                }
        except:
            crypto_data[name] = {
                'current': 0,
                'change': 0,
                'percent_change': 0,
                'week_change': 0
            }
    return crypto_data

# Portfolio metrics
def calculate_portfolio_metrics(df, current_prices):
    df = df.copy()
    df['symbol'] = df['Stock']
    df['shares'] = df['Shares']
    df['purchase_price'] = df['Purchase Price']
    df['holding_type'] = df['Term'].apply(lambda x: 'long_term' if x.lower() == 'long' else 'short_term')
    df['current_price'] = df['symbol'].map(current_prices)
    df['current_value'] = df['shares'] * df['current_price']
    df['purchase_value'] = df['shares'] * df['purchase_price']
    df['unrealized_gain'] = df['current_value'] - df['purchase_value']
    df['unrealized_gain_percent'] = (df['unrealized_gain'] / df['purchase_value']) * 100
    df['tax_rate'] = df['holding_type'].apply(lambda x: 0.15 if x == 'long_term' else 0.35)
    df['estimated_tax'] = df.apply(lambda x: x['unrealized_gain'] * x['tax_rate'] if x['unrealized_gain'] > 0 else 0, axis=1)
    df['effective_tax_rate'] = df.apply(lambda x: (x['estimated_tax'] / x['unrealized_gain'] * 100) if x['unrealized_gain'] > 0 else 0, axis=1)
    return df
# Initialize OpenAI client
def get_openai_client():
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = ""
    return OpenAI(api_key=st.session_state.OPENAI_API_KEY)

# Generate AI chat response
def generate_chat_response(query, portfolio_data, risk_tolerance, tax_sensitivity, investment_horizon, openai_client):
    try:
        if not openai_client.api_key:
            return "⚠️ Please enter your OpenAI API key in the Settings tab to enable chat."

        symbols = portfolio_data['Stock'].tolist()
        current_prices = get_current_prices(symbols)
        metrics = calculate_portfolio_metrics(portfolio_data, current_prices)

        total_value = metrics['current_value'].sum()
        total_gain = metrics['unrealized_gain'].sum()
        total_gain_pct = (total_gain / metrics['purchase_value'].sum()) * 100 if metrics['purchase_value'].sum() else 0

        top = metrics.loc[metrics['unrealized_gain_percent'].idxmax()]
        worst = metrics.loc[metrics['unrealized_gain_percent'].idxmin()]

        context = f"""
        Portfolio Value: ${total_value:.2f}
        Total Gain: ${total_gain:.2f} ({total_gain_pct:.2f}%)
        Top Performer: {top['symbol']} ({top['unrealized_gain_percent']:.2f}%)
        Worst Performer: {worst['symbol']} ({worst['unrealized_gain_percent']:.2f}%)

        Risk: {risk_tolerance}, Tax Sensitivity: {tax_sensitivity}, Horizon: {investment_horizon}
        """

        prompt = f"""
        You are DAJANIII, an expert portfolio advisor. Here's the context:

        {context}

        Answer the user's question:
        {query}
        """

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an intelligent stock assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
# Display header
display_header()

# Sidebar
with st.sidebar:
    st.header("📂 Upload Portfolio")
    uploaded_csv = st.file_uploader("Upload Schwab CSV", type=["csv"])
    uploaded_pdf = st.file_uploader("Upload IB PDF", type=["pdf"])
    if st.button("Use Sample Portfolio"):
        st.session_state.portfolio_df = load_sample_portfolio()

    st.header("📊 Investor Profile")
    risk_tolerance = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1)
    tax_sensitivity = st.selectbox("Tax Sensitivity", ["low", "moderate", "high"], index=1)
    investment_horizon = st.selectbox("Investment Horizon", ["short", "medium", "long"], index=1)

    st.header("🔐 OpenAI Key")
    api_key = st.text_input("OpenAI API Key", value=st.session_state.get("OPENAI_API_KEY", ""), type="password")
    if api_key:
        st.session_state.OPENAI_API_KEY = api_key

# Load uploaded files
if uploaded_csv:
    try:
        content = uploaded_csv.getvalue().decode('utf-8')
        st.session_state.portfolio_df = parse_schwab_portfolio(content)
    except Exception as e:
        st.error(f"CSV Error: {e}")

if uploaded_pdf:
    try:
        content = uploaded_pdf.getvalue()
        st.session_state.portfolio_df = parse_ib_portfolio(content)
    except Exception as e:
        st.error(f"PDF Error: {e}")

# Fallback if no upload
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = None

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Chat Interface
st.header("💬 Chat with DAJANIII")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your portfolio, stocks, or market trends..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.portfolio_df is None:
        response = "Please upload a portfolio first."
    else:
        client = get_openai_client()
        response = generate_chat_response(
            prompt,
            st.session_state.portfolio_df,
            risk_tolerance,
            tax_sensitivity,
            investment_horizon,
            client
        )

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
# --- Portfolio Summary Dashboard ---
if st.session_state.portfolio_df is not None:
    df = st.session_state.portfolio_df
    symbols = df['Stock'].tolist()
    prices = get_current_prices(symbols)
    metrics = calculate_portfolio_metrics(df, prices)

    st.subheader("📈 Portfolio Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"${metrics['current_value'].sum():,.2f}")
    col2.metric("Unrealized Gain", f"${metrics['unrealized_gain'].sum():,.2f}")
    col3.metric("Estimated Tax", f"${metrics['estimated_tax'].sum():,.2f}")

    st.dataframe(metrics[['symbol', 'shares', 'purchase_price', 'current_price',
                          'unrealized_gain_percent', 'estimated_tax']])

    # Sector Pie Chart
    st.subheader("📊 Sector Allocation")
    sector_data = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            sector = info.get("sector", "Unknown")
            sector_data[sector] = sector_data.get(sector, 0) + 1
        except:
            sector_data["Unknown"] = sector_data.get("Unknown", 0) + 1

    if sector_data:
        fig = px.pie(
            names=list(sector_data.keys()),
            values=list(sector_data.values()),
            title="Sectors"
        )
        st.plotly_chart(fig)

# --- Market Overview ---
st.subheader("📉 Market Index Overview")
indexes = get_market_data()
cols = st.columns(len(indexes))
for i, (name, data) in enumerate(indexes.items()):
    change_str = f"{data['change']:+.2f} ({data['percent_change']:+.2f}%)"
    with cols[i]:
        st.metric(name, f"{data['current']:.2f}", change_str)

# --- Crypto Dashboard ---
st.subheader("₿ Crypto Market")
crypto = get_crypto_data()
crypto_cols = st.columns(len(crypto))
for i, (name, data) in enumerate(crypto.items()):
    daily = f"{data['percent_change']:+.2f}%"
    weekly = f"{data['week_change']:+.2f}%"
    with crypto_cols[i]:
        st.metric(name, f"${data['current']:.2f}", daily)
        st.caption(f"7d: {weekly}")
# --- Footer ---
st.markdown("""<hr style="margin-top: 40px;">""", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "DAJANIII • AI-Powered Stock Trading Assistant<br>"
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>", unsafe_allow_html=True
)
