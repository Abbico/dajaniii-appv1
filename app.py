
import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(page_title="DAJANIII Assistant", layout="wide")

st.title("📊 DAJANIII Stock Trading Assistant")

# Settings
st.sidebar.header("Settings")
use_openrouter = st.sidebar.radio("Use which AI Provider?", ["OpenRouter", "OpenAI"], index=0)

if use_openrouter == "OpenRouter":
    openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")
else:
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

@st.cache_data(ttl=300)
def get_current_prices(symbols):
    prices = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            prices[symbol] = hist['Close'].iloc[-1] if not hist.empty else 0
        except:
            prices[symbol] = 0
    return prices

def calculate_portfolio_metrics(df, current_prices):
    df = df.copy()
    df['symbol'] = df['Stock']
    df['shares'] = df['Shares']
    df['purchase_price'] = df['Purchase Price']
    df['current_price'] = df['symbol'].map(current_prices)
    df['current_value'] = df['shares'] * df['current_price']
    df['purchase_value'] = df['shares'] * df['purchase_price']
    df['unrealized_gain'] = df['current_value'] - df['purchase_value']
    df['unrealized_gain_percent'] = (df['unrealized_gain'] / df['purchase_value']) * 100
    return df

def generate_openrouter_response(query, df, risk, tax, horizon, openrouter_key):
    try:
        prices = get_current_prices(df['Stock'])
        metrics = calculate_portfolio_metrics(df, prices)
        summary = (
            f"Portfolio value: ${metrics['current_value'].sum():,.2f}\n"
            f"Total gain: ${metrics['unrealized_gain'].sum():,.2f}\n"
            f"Top performer: {metrics.iloc[metrics['unrealized_gain_percent'].idxmax()]['symbol']} "
            f"({metrics.iloc[metrics['unrealized_gain_percent'].idxmax()]['unrealized_gain_percent']:.2f}%)\n"
            f"Worst performer: {metrics.iloc[metrics['unrealized_gain_percent'].idxmin()]['symbol']} "
            f"({metrics.iloc[metrics['unrealized_gain_percent'].idxmin()]['unrealized_gain_percent']:.2f}%)"
        )

        prompt = f"""
        You are a financial assistant. User profile:
        Risk: {risk}
        Tax Sensitivity: {tax}
        Investment Horizon: {horizon}

        Portfolio Summary:
        {summary}

        Answer the question: {query}
        """

        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a smart financial assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ OpenRouter Error: {str(e)}"

# Sample portfolio
sample_portfolio = pd.DataFrame({
    'Stock': ['AAPL', 'MSFT', 'GOOGL'],
    'Shares': [10, 5, 3],
    'Purchase Price': [150, 200, 1800]
})

st.subheader("📁 Portfolio Preview")
st.dataframe(sample_portfolio)

query = st.text_input("Ask a question about your portfolio...")

if query:
    if use_openrouter == "OpenRouter":
        if not openrouter_key:
            st.warning("Please enter your OpenRouter API Key.")
        else:
            st.info("Using OpenRouter...")
            response = generate_openrouter_response(query, sample_portfolio, "moderate", "high", "long", openrouter_key)
            st.markdown(response)
    else:
        st.warning("OpenAI toggle selected, but OpenAI support is not implemented in this version.")
