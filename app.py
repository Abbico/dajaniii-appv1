
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import io
import re
from pypdf import PdfReader

st.set_page_config(page_title="📊 DAJANIII v2", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .title-container {
        text-align: center;
        background-color: #ffcc00;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .title-container h1 {
        color: #2f2f2f;
    }
    .portfolio-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    </style>
    <div class="title-container">
        <h1>📈 DAJANIII Portfolio Playground</h1>
        <p>Upload your portfolio PDFs and watch the magic!</p>
    </div>
""", unsafe_allow_html=True)

# Store multiple portfolios in session
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = []

pdf = st.file_uploader("📄 Upload any Portfolio PDF", type="pdf")

def extract_pdf_data(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for p in reader.pages:
            text += p.extract_text()
        entries = re.findall(r"([A-Z]{2,5})\s+(\d[\d,]*)\s+(\d+\.\d+)", text)
        return pd.DataFrame([{
            'Stock': symbol,
            'Shares': float(shares.replace(",", "")),
            'Purchase Price': float(price),
        } for symbol, shares, price in entries])
    except:
        return pd.DataFrame()

if pdf:
    df = extract_pdf_data(pdf.read())
    if not df.empty:
        df['Term'] = 'Long'
        df['id'] = f"Portfolio {len(st.session_state.portfolios)+1}"
        st.session_state.portfolios.append(df)
        st.success(f"✅ Uploaded {df.shape[0]} positions to Portfolio {len(st.session_state.portfolios)}")

# Combine all portfolios
if st.session_state.portfolios:
    all_df = pd.concat(st.session_state.portfolios)
    grouped_df = all_df.groupby(['Stock'], as_index=False).agg({
        'Shares': 'sum',
        'Purchase Price': 'mean',
        'Term': 'first'
    })

    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.subheader("📚 Combined Portfolio")
    st.dataframe(grouped_df)

    symbols = grouped_df['Stock'].tolist()

    @st.cache_data(ttl=300)
    def get_prices(symbols):
        results = {}
        for s in symbols:
            try:
                results[s] = yf.Ticker(s).history(period="1d")['Close'].iloc[-1]
            except:
                results[s] = 0
        return results

    prices = get_prices(symbols)
    grouped_df['Current Price'] = grouped_df['Stock'].map(prices)
    grouped_df['Current Value'] = grouped_df['Shares'] * grouped_df['Current Price']
    grouped_df['Purchase Value'] = grouped_df['Shares'] * grouped_df['Purchase Price']
    grouped_df['Gain %'] = ((grouped_df['Current Value'] - grouped_df['Purchase Value']) / grouped_df['Purchase Value']) * 100

    st.metric("💰 Total Value", f"${grouped_df['Current Value'].sum():,.2f}")
    st.metric("📈 Gain/Loss", f"{grouped_df['Gain %'].mean():.2f}%")

    # Pie chart
    st.subheader("🍩 Portfolio Allocation")
    fig = px.pie(grouped_df, names='Stock', values='Current Value', title="By Market Value")
    st.plotly_chart(fig, use_container_width=True)

    # Bar comparison with index
    st.subheader("📊 Compare with Index")
    index_symbol = st.selectbox("Compare against:", ['^GSPC', '^IXIC', '^DJI'], index=0)
    try:
        index_data = yf.Ticker(index_symbol).history(period="30d")
        stock_data = {}
        for stock in symbols:
            hist = yf.Ticker(stock).history(period="30d")
            if not hist.empty:
                stock_data[stock] = hist['Close']

        if index_data is not None:
            df_plot = pd.DataFrame(stock_data)
            df_plot['Index'] = index_data['Close'].values[:len(df_plot)]
            st.line_chart(df_plot)
    except Exception as e:
        st.warning(f"Unable to fetch comparison data: {e}")
