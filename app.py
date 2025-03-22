import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from openai import OpenAI
import io

# --- Page Config ---
st.set_page_config(page_title="DAJANIII - AI Stock Assistant", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .dajaniii-header {
        background-color: #2E7D32; color: white;
        padding: 20px; text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: white; padding: 15px;
        border-radius: 10px; box-shadow: 0 0 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
</style>
"", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="dajaniii-header"><h1>DAJANIII</h1><h4>AI-Powered Stock Assistant</h4></div>', unsafe_allow_html=True)

# --- Load Sample Portfolio ---
def load_sample_portfolio():
    data = {
        'Stock': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN'],
        'Shares': [10, 15, 5, 8, 6],
        'Purchase Price': [150, 300, 1200, 500, 3100]
    }
    return pd.DataFrame(data)

# --- Parse Schwab CSV ---
def parse_schwab_portfolio(content):
    df = pd.read_csv(io.StringIO(content), skiprows=3)
    df.columns = [col.split('(')[0].strip() for col in df.columns]
    df = df.rename(columns={'Symbol': 'Stock', 'Qty': 'Shares', 'Price': 'Purchase Price'})
    df = df[['Stock', 'Shares', 'Purchase Price']]
    return df

# --- Get Prices ---
@st.cache_data(ttl=300)
def get_current_prices(symbols):
    prices = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="1d")
            prices[sym] = hist['Close'].iloc[-1] if not hist.empty else 0
        except:
            prices[sym] = 0
    return prices

# --- Portfolio Metrics ---
def calculate_portfolio_metrics(df, prices):
    df = df.copy()
    df['symbol'] = df['Stock']
    df['shares'] = df['Shares']
    df['purchase_price'] = df['Purchase Price']
    df['current_price'] = df['symbol'].map(prices)
    df['current_value'] = df['shares'] * df['current_price']
    df['purchase_value'] = df['shares'] * df['purchase_price']
    df['unrealized_gain'] = df['current_value'] - df['purchase_value']
    df['unrealized_gain_percent'] = df['unrealized_gain'] / df['purchase_value'] * 100
    df['tax_rate'] = 0.15
    df['estimated_tax'] = df['unrealized_gain'] * df['tax_rate']
    return df

# --- OpenAI API Setup ---
def get_openai_client():
    if 'OPENAI_API_KEY' not in st.session_state:
        st.session_state.OPENAI_API_KEY = ""
    return OpenAI(api_key=st.session_state.OPENAI_API_KEY)

# --- Chat Response ---
def generate_chat_response(query, df, risk, tax, horizon, client):
    try:
        prices = get_current_prices(df['Stock'])
        metrics = calculate_portfolio_metrics(df, prices)
        summary = f"Portfolio value: ${metrics['current_value'].sum():,.2f}\n"
        summary += f"Total gain: ${metrics['unrealized_gain'].sum():,.2f}\n"
        top = metrics.iloc[metrics['unrealized_gain_percent'].idxmax()]
        worst = metrics.iloc[metrics['unrealized_gain_percent'].idxmin()]
        summary += f"Top performer: {top['symbol']} ({top['unrealized_gain_percent']:.2f}%)\n"
        summary += f"Worst performer: {worst['symbol']} ({worst['unrealized_gain_percent']:.2f}%)\n"

        prompt = f"""You are a financial assistant. User profile: Risk={risk}, Tax={tax}, Horizon={horizon}.
        Portfolio Summary:
        {summary}
        Answer the question: {query}"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a smart financial assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as api_error:
            if "insufficient_quota" in str(api_error):
                return "⚠️ Your OpenAI quota has been exceeded. Visit https://platform.openai.com/account/billing."
            else:
                return f"🚨 OpenAI API Error: {api_error}"
    except Exception as e:
        return f"❌ Error: {e}"

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Portfolio Upload")
    csv_file = st.file_uploader("Upload Schwab CSV", type=["csv"])
    sample = st.button("Use Sample Portfolio")
    st.header("💼 Investor Profile")
    risk = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1)
    tax = st.selectbox("Tax Sensitivity", ["low", "moderate", "high"], index=1)
    horizon = st.selectbox("Investment Horizon", ["short", "medium", "long"], index=1)
    st.header("🔑 OpenAI API Key")
    key = st.text_input("Paste API Key", type="password")
    if key: st.session_state.OPENAI_API_KEY = key

# --- Load Portfolio ---
if sample:
    st.session_state.df = load_sample_portfolio()
elif csv_file:
    content = csv_file.getvalue().decode("utf-8")
    st.session_state.df = parse_schwab_portfolio(content)

# --- Main App ---
st.header("💬 Ask DAJANIII")
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your portfolio or stocks..."):
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "df" not in st.session_state:
        response = "⚠️ Please upload a portfolio or use a sample."
    else:
        client = get_openai_client()
        response = generate_chat_response(prompt, st.session_state.df, risk, tax, horizon, client)

    with st.chat_message("assistant"): st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Portfolio Metrics View ---
if "df" in st.session_state:
    df = st.session_state.df
    st.subheader("📊 Portfolio Metrics")
    prices = get_current_prices(df['Stock'])
    metrics = calculate_portfolio_metrics(df, prices)

    col1, col2, col3 = st.columns(3)
    col1.metric("Value", f"${metrics['current_value'].sum():,.2f}")
    col2.metric("Unrealized Gain", f"${metrics['unrealized_gain'].sum():,.2f}")
    col3.metric("Est. Tax", f"${metrics['estimated_tax'].sum():,.2f}")

    st.dataframe(metrics[['symbol', 'shares', 'purchase_price', 'current_price',
                          'unrealized_gain_percent', 'estimated_tax']])

# --- Footer ---
st.markdown("<hr><center style='color:gray'>DAJANIII • Stock AI Assistant<br>Updated: {}</center>".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
