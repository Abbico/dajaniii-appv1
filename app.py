
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import io
from datetime import datetime
from pypdf import PdfReader
import re

st.set_page_config(page_title="DAJANIII", layout="wide")

# Hidden OpenRouter key
_default_openrouter_key = "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"

# Logo header
st.markdown("""
    <div style='text-align: center; padding: 1rem; background: #1c1c1c; color: white; border-radius: 10px;'>
        <img src='logo.png' width='100'><h1>DAJANIII</h1><p>AI-Powered Stock Assistant</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("API Provider")
    api_provider = st.radio("Choose", ["OpenRouter", "OpenAI"])
    api_key = st.text_input("API Key", value="", type="password", placeholder="Optional override")
    if api_provider == "OpenRouter" and not api_key:
        api_key = _default_openrouter_key

    st.markdown("---")
    st.header("Upload Portfolio")
    csv = st.file_uploader("Upload Schwab CSV", type="csv")
    pdf = st.file_uploader("Upload IBKR PDF", type="pdf")

# Default example portfolio
df = pd.DataFrame({'Stock': ['AAPL', 'MSFT'], 'Shares': [10, 5], 'Purchase Price': [150, 250], 'Term': ['Long', 'Short']})

if csv:
    try:
        raw = csv.read().decode("utf-8")
        df_csv = pd.read_csv(io.StringIO(raw), skiprows=3)
        df_csv.columns = [c.split('(')[0].strip() for c in df_csv.columns]
        df_csv = df_csv.rename(columns={'Symbol': 'Stock', 'Qty': 'Shares', 'Price': 'Purchase Price'})
        df = df_csv[['Stock', 'Shares', 'Purchase Price']].dropna()
        df['Term'] = 'Long'
    except Exception as e:
        st.error(f"CSV Error: {e}")

elif pdf:
    try:
        reader = PdfReader(io.BytesIO(pdf.read()))
        content = ""
        for p in reader.pages:
            content += p.extract_text()
        rows = re.findall(r"([A-Z]+)\s+([\d,]+)\s+([\d.]+)", content)
        df = pd.DataFrame([{
            'Stock': s,
            'Shares': float(q.replace(",", "")),
            'Purchase Price': float(p),
            'Term': 'Long'
        } for s, q, p in rows])
    except Exception as e:
        st.error(f"PDF Error: {e}")

@st.cache_data(ttl=300)
def prices(symbols):
    out = {s: 0 for s in symbols}
    for s in symbols:
        try:
            out[s] = yf.Ticker(s).history(period="1d")['Close'].iloc[-1]
        except:
            continue
    return out

if not df.empty:
    st.subheader("📈 Portfolio Overview")
    live = prices(df['Stock'])
    df['Current Price'] = df['Stock'].map(live)
    df['Value'] = df['Shares'] * df['Current Price']
    df['Purchase'] = df['Shares'] * df['Purchase Price']
    df['Gain %'] = (df['Value'] - df['Purchase']) / df['Purchase'] * 100

    st.dataframe(df, use_container_width=True)

    total = df['Value'].sum()
    gain = total - df['Purchase'].sum()
    st.metric("Total Value", f"${{total:,.2f}}")
    st.metric("Total Gain/Loss", f"${{gain:,.2f}}")

    st.subheader("📊 Allocation")
    fig = px.pie(df, values='Value', names='Stock', title='Portfolio Allocation')
    st.plotly_chart(fig, use_container_width=True)
