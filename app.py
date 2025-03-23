import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
import pdfplumber
import re
import numpy as np
import pandas_ta as ta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Set page configuration with a fun title
st.set_page_config(
    page_title="DAJANII Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more cartoonish, fun style
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --accent-color: #FFD166;
        --background-color: #f0f8ff;
        --text-color: #2A2A72;
    }
    
    /* Background and text */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--primary-color);
        font-family: 'Comic Sans MS', cursive, sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--secondary-color);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        border: 2px solid white;
        box-shadow: 3px 3px 5px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
    }
    
    /* Cards for content sections */
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 5px 5px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    .stMetric {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 3px 3px 5px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2A2A72;
    }
    
    .sidebar .sidebar-content {
        background-color: #2A2A72;
    }
    
    /* Upload button */
    .uploadButton {
        background-color: var(--accent-color);
        padding: 20px;
        border-radius: 15px;
        border: 3px dashed #FF6B6B;
        text-align: center;
    }
    
    /* Fun animations */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: var(--primary-color);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Function to get base64 encoded image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Logo and header with animation
logo_base64 = get_base64_encoded_image("/home/ubuntu/portfolio_app/logo.png")

st.markdown(f"""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #2A2A72, #009FFD); color: white; border-radius: 20px; margin-bottom: 20px;' class='floating'>
    <img src='data:image/png;base64,{logo_base64}' width='120'>
    <h1 style='font-size: 3em; margin-top: 10px;'>DAJANII</h1>
    <p style='font-size: 1.5em; font-family: "Comic Sans MS", cursive, sans-serif;'>AI-Powered Stock Portfolio Analyzer</p>
</div>
""", unsafe_allow_html=True)

# Create a container for the main content
main_container = st.container()

# Sidebar configuration
with st.sidebar:
    st.markdown("<h2 style='color: white; font-family: \"Comic Sans MS\", cursive, sans-serif;'>🛠️ Tools & Settings</h2>", unsafe_allow_html=True)
    
    # API settings
    st.markdown("<div style='background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>API Provider</h3>", unsafe_allow_html=True)
    api_provider = st.radio("Choose", ["OpenRouter", "OpenAI"], key="api_provider")
    api_key = st.text_input("API Key", value="", type="password", placeholder="Optional override")
    
    # Hidden OpenRouter key
    _default_openrouter_key = "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"
    if api_provider == "OpenRouter" and not api_key:
        api_key = _default_openrouter_key
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Portfolio settings
    st.markdown("<div style='background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>Analysis Settings</h3>", unsafe_allow_html=True)
    
    # Time period for analysis
    time_period = st.select_slider(
        "Analysis Time Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        value="1y"
    )
    
    # Comparison index
    comparison_index = st.multiselect(
        "Compare with Indexes",
        ["^GSPC (S&P 500)", "^DJI (Dow Jones)", "^IXIC (NASDAQ)", "^RUT (Russell 2000)"],
        default=["^GSPC (S&P 500)"]
    )
    
    # Technical indicators
    st.markdown("<h4 style='color: white;'>Technical Indicators</h4>", unsafe_allow_html=True)
    
    tech_indicators = st.multiselect(
        "Select Indicators",
        ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "Stochastic Oscillator"],
        default=["SMA", "RSI"]
    )
    
    # Prediction settings
    st.markdown("<h4 style='color: white;'>Prediction Settings</h4>", unsafe_allow_html=True)
    prediction_days = st.slider("Prediction Days", min_value=7, max_value=90, value=30, step=7)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main content
with main_container:
    # Portfolio upload section with fun styling
    st.markdown("""
    <div class='card'>
        <h2>📊 Upload Your Portfolio</h2>
        <p>Upload any portfolio PDF file to analyze your investments!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Single upload area for any PDF
    uploaded_file = st.file_uploader("Upload Portfolio PDF", type="pdf", help="Upload any portfolio PDF to extract holdings data")
    
    # Portfolio storage
    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = {}
        
    if 'combined_portfolio' not in st.session_state:
        st.session_state.combined_portfolio = pd.DataFrame()
    
    # Process uploaded PDF
    if uploaded_file:
        try:
            # Create a unique name for this portfolio based on upload time
            portfolio_name = f"Portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Extract data from PDF
            with st.spinner("Extracting data from PDF..."):
                # Use pdfplumber for better text extraction
                pdf = pdfplumber.open(uploaded_file)
                content = ""
                
                for page in pdf.pages:
                    content += page.extract_text() + "\n"
                
                # More robust pattern matching for various portfolio formats
                # Look for patterns like stock symbols (1-5 uppercase letters) followed by numbers
                # This regex looks for stock symbols and their associated quantities and prices
                rows = []
                
                # Try different regex patterns to match various portfolio formats
                patterns = [
                    # Pattern for "SYMBOL Quantity Price" format
                    r"([A-Z]{1,5})\s+([\d,]+)\s+([\d,.]+)",
                    # Pattern for "SYMBOL - Quantity @ Price" format
                    r"([A-Z]{1,5})\s*[-–]\s*([\d,]+)\s*[@at]\s*([\d,.]+)",
                    # Pattern for "Stock: SYMBOL, Shares: Quantity, Price: Price" format
                    r"Stock:?\s*([A-Z]{1,5}).*?Shares:?\s*([\d,]+).*?Price:?\s*\$?([\d,.]+)"
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        for match in matches:
                            symbol, quantity, price = match
                            rows.append({
                                'Stock': symbol.strip(),
                                'Shares': float(quantity.replace(",", "")),
                                'Purchase Price': float(price.replace(",", "")),
                                'Term': 'Long'  # Default to long-term holdings
                            })
                        break  # Stop if we found matches with this pattern
                
                # If no matches found with predefined patterns, try a more general approach
                if not rows:
                    # Look for potential stock symbols (1-5 uppercase letters)
                    potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', content)
                    
                    for symbol in potential_symbols:
                        # Look for numbers near the symbol that could be quantities and prices
                        symbol_context = re.search(f"{symbol}.*?(\d+[\d,.]*?).*?(\d+[\d,.]*)", content)
                        if symbol_context:
                            quantity, price = symbol_context.groups()
                            rows.append({
                                'Stock': symbol,
                                'Shares': float(quantity.replace(",", "")),
                                'Purchase Price': float(price.replace(",", "")),
                                'Term': 'Long'
                            })
                
                # Create DataFrame from extracted data
                if rows:
                    df = pd.DataFrame(rows)
                    
                    # Store this portfolio in session state
                    st.session_state.portfolios[portfolio_name] = df
                    
                    # Update combined portfolio
                    update_combined_portfolio()
                    
                    st.success(f"Successfully extracted {len(df)} positions from the PDF!")
                else:
                    st.error("Could not extract portfolio data from the PDF. Please check the format.")
                    
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    
    # Function to update the combined portfolio
    def update_combined_portfolio():
        if st.session_state.portfolios:
            # Combine all portfolios
            combined = pd.concat([df for df in st.session_state.portfolios.values()])
            
            # Group by stock to combine positions
            combined = combined.groupby('Stock').agg({
                'Shares': 'sum',
                'Purchase Price': lambda x: np.average(x, weights=combined.loc[x.index, 'Shares']),
                'Term': lambda x: 'Long' if 'Long' in x.values else 'Short'
            }).reset_index()
            
            st.session_state.combined_portfolio = combined
    
    # Display portfolios if available
    if st.session_state.portfolios:
        st.markdown("""
        <div class='card'>
            <h2>🗂️ Your Portfolios</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for individual portfolios and combined view
        tabs = ["Combined Portfolio"] + list(st.session_state.portfolios.keys())
        selected_tab = st.tabs(tabs)
        
        # Function to get current prices and calculate metrics
        @st.cache_data(ttl=300)
        def get_stock_prices(symbols):
            prices = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    prices[symbol] = current_price
                except Exception as e:
                    st.warning(f"Could not fetch price for {symbol}: {str(e)}")
                    prices[symbol] = 0
            return prices
        
        # Combined Portfolio Tab
        with selected_tab[0]:
            if not st.session_state.combined_portfolio.empty:
                df = st.session_state.combined_portfolio.copy()
                
                # Get current prices
                current_prices = get_stock_prices(df['Stock'].unique())
                
                # Calculate portfolio metrics
                df['Current Price'] = df['Stock'].map(current_prices)
                df['Current Value'] = df['Shares'] * df['Current Price']
                df['Purchase Value'] = df['Shares'] * df['Purchase Price']
                df['Gain/Loss'] = df['Current Value'] - df['Purchase Value']
                df['Gain %'] = (df['Current Value'] - df['Purchase Value']) / df['Purchase Value'] * 100
                
                # Display portfolio metrics in a fun, colorful way
                col1, col2, col3 = st.columns(3)
                
                total_value = df['Current Value'].sum()
                total_cost = df['Purchase Value'].sum()
                total_gain = total_value - total_cost
                total_gain_pct = (total_gain / total_cost) * 100 if total_cost > 0 else 0
                
                gain_color = "green" if total_gain >= 0 else "red"
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                        <h3 style='margin: 0; color: #2A2A72;'>Total Value</h3>
                        <p style='font-size: 24px; font-weight: bold; margin: 10px 0;'>${total_value:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                        <h3 style='margin: 0; color: #2A2A72;'>Total Gain/Loss</h3>
                        <p style='font-size: 24px; font-weight: bold; margin: 10px 0; color: {gain_color};'>${total_gain:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                        <h3 style='margin: 0; color: #2A2A72;'>Return</h3>
                        <p style='font-size: 24px; font-weight: bold; margin: 10px 0; color: {gain_color};'>{total_gain_pct:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display the portfolio table with styling
                st.markdown("<h3>Holdings</h3>", unsafe_allow_html=True)
                
                # Format the dataframe for display
                display_df = df.copy()
                display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
                display_df['Purchase Price'] = display_df['Purchase Price'].map('${:,.2f}'.format)
                display_df['Current Value'] = display_df['Current Value'].map('${:,.2f}'.format)
                display_df['Purchase Value'] = display_df['Purchase Value'].map('${:,.2f}'.format)
                display_df['Gain/Loss'] = display_df['Gain/Loss'].map('${:,.2f}'.format)
                display_df['Gain %'] = display_df['Gain %'].map('{:,.2f}%'.format)
                
                st.dataframe(
                    display_df,
                    column_config={
                        "Stock": st.column_config.TextColumn("Stock Symbol"),
                        "Shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                        "Purchase Price": st.column_config.TextColumn("Purchase Price"),
                        "Current Price": st.column_config.TextColumn("Current Price"),
                        "Purchase Value": st.column_config.TextColumn("Purchase Value"),
                        "Current Value": st.column_config.TextColumn("Current Value"),
                        "Gain/Loss": st.column_config.TextColumn("Gain/Loss"),
                        "Gain %": st.column_config.TextColumn("Return %"),
                        "Term": st.column_config.TextColumn("Term")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Portfolio Allocation Chart
                st.markdown("<h3>Portfolio Allocation</h3>", unsafe_allow_html=True)
                
                # Create a more colorful and fun pie chart
                fig = px.pie(
                    df, 
                    values='Current Value', 
                    names='Stock',
                    title='Portfolio Allocation by Value',
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    hole=0.4
                )
                
                # Update layout for a more fun look
                fig.update_layout(
                    font=dict(family="Comic Sans MS, cursive, sans-serif", size=14),
                    title_font=dict(family="Comic Sans MS, cursive, sans-serif", size=20),
                    legend_title_font=dict(family="Comic Sans MS, cursive, sans-serif"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=50, b=50, l=20, r=20)
                )
                
                # Add annotations in the center
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#FFFFFF', width=2))
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance Comparison with Indexes
                st.markdown("<h3>Performance Comparison</h3>", unsafe_allow_html=True)
                
                # Get top 5 holdings by value for comparison
                top_holdings = df.sort_values('Current Value', ascending=False).head(5)['Stock'].tolist()
                
                # Get comparison indexes
                comparison_symbols = [idx.split(" ")[0] for idx in comparison_index]
                
                # Combine top holdings and indexes for comparison
                comparison_symbols = top_holdings + comparison_symbols
                
                # Get historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)  # Default to 1 year
                
                if time_period == "1mo":
                    start_date = end_date - timedelta(days=30)
                elif time_period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif time_period == "6mo":
                    start_date = end_date - timedelta(days=180)
                elif time_period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif time_period == "2y":
                    start_date = end_date - timedelta(days=730)
                elif time_period == "5y":
                    start_date = end_date - timedelta(days=1825)
                
                # Get historical data
                historical_data = {}
                for symbol in comparison_symbols:
                    try:
                        data = yf.download(symbol, start=start_date, end=end_date)
                        if not data.empty:
                            # Normalize to 100 at the beginning
                            first_price = data['Close'].iloc[0]
                            historical_data[symbol] = (data['Close'] / first_price) * 100
                    except Exception as e:
                        st.warning(f"Could not fetch historical data for {symbol}: {str(e)}")
                
                # Create comparison chart
                if historical_data:
                    # Convert to DataFrame
                    hist_df = pd.DataFrame(historical_data)
                    
                    # Create a more colorful line chart
                    fig = go.Figure()
                    
                    # Add traces for top holdings
                    for symbol in top_holdings:
                        if symbol in hist_df.columns:
                            fig.add_trace(go.Scatter(
                                x=hist_df.index,
                                y=hist_df[symbol],
                                mode='lines',
                                name=symbol,
                                line=dict(width=3),
                                hovertemplate='%{y:.2f}%<extra></extra>'
                            ))
                    
                    # Add traces for indexes with dashed lines
                    for idx in comparison_symbols:
                        if idx in hist_df.columns and idx not in top_holdings:
                            fig.add_trace(go.Scatter(
                                x=hist_df.index,
                                y=hist_df[idx],
                                mode='lines',
                                name=idx,
                                line=dict(width=2, dash='dash'),
                                hovertemplate='%{y:.2f}%<extra></extra>'
                            ))
                    
                    # Update layout for a more fun look
                    fig.update_layout(
                        title=f"Performance Comparison (Normalized to 100) - {time_period}",
                        xaxis_title="Date",
                        yaxis_title="Normalized Price (%)",
                        font=dict(family="Comic Sans MS, cursive, sans-serif"),
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.03)',
                        margin=dict(t=80, b=50, l=20, r=20)
                    )
                    
                    # Add grid lines
                    fig.update_xaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(0,0,0,0.1)'
                    )
                    
                    fig.update_yaxes(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(0,0,0,0.1)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Technical Analysis Section
                st.markdown("<h3>Technical Analysis</h3>", unsafe_allow_html=True)
                
                # Select a stock for technical analysis
                selected_stock = st.selectbox("Select a stock for technical analysis", top_holdings)
                
                if selected_stock:
                    # Get historical data for the selected stock
                    stock_data = yf.download(selected_stock, start=start_date, end=end_date)
                    
                    if not stock_data.empty:
                        # Create a DataFrame for technical analysis
                        ta_df = stock_data.copy()
                        
                        # Calculate technical indicators based on user selection
                        if "SMA" in tech_indicators:
                            ta_df['SMA_20'] = ta_df['Close'].rolling(window=20).mean()
                            ta_df['SMA_50'] = ta_df['Close'].rolling(window=50).mean()
                            ta_df['SMA_200'] = ta_df['Close'].rolling(window=200).mean()
                        
                        if "EMA" in tech_indicators:
                            ta_df['EMA_12'] = ta_df['Close'].ewm(span=12, adjust=False).mean()
                            ta_df['EMA_26'] = ta_df['Close'].ewm(span=26, adjust=False).mean()
                        
                        if "RSI" in tech_indicators:
                            ta_df['RSI'] = ta.rsi(ta_df['Close'], length=14)
                        
                        if "MACD" in tech_indicators:
                            macd = ta.macd(ta_df['Close'])
                            ta_df = pd.concat([ta_df, macd], axis=1)
                        
                        if "Bollinger Bands" in tech_indicators:
                            bollinger = ta.bbands(ta_df['Close'], length=20, std=2)
                            ta_df = pd.concat([ta_df, bollinger], axis=1)
                        
                        if "Stochastic Oscillator" in tech_indicators:
                            stoch = ta.stoch(ta_df['High'], ta_df['Low'], ta_df['Close'], k=14, d=3, smooth_k=3)
                            ta_df = pd.concat([ta_df, stoch], axis=1)
                        
                        # Create technical analysis chart
                        fig = go.Figure()
                        
                        # Add price candlestick
                        fig.add_trace(go.Candlestick(
                            x=ta_df.index,
                            open=ta_df['Open'],
                            high=ta_df['High'],
                            low=ta_df['Low'],
                            close=ta_df['Close'],
                            name="Price",
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        ))
                        
                        # Add technical indicators
                        if "SMA" in tech_indicators:
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['SMA_20'],
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='blue', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['SMA_50'],
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='orange', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['SMA_200'],
                                mode='lines',
                                name='SMA 200',
                                line=dict(color='purple', width=1)
                            ))
                        
                        if "EMA" in tech_indicators:
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['EMA_12'],
                                mode='lines',
                                name='EMA 12',
                                line=dict(color='green', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['EMA_26'],
                                mode='lines',
                                name='EMA 26',
                                line=dict(color='red', width=1)
                            ))
                        
                        if "Bollinger Bands" in tech_indicators and 'BBL_20_2.0' in ta_df.columns:
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['BBU_20_2.0'],
                                mode='lines',
                                name='Upper BB',
                                line=dict(color='rgba(0,0,255,0.5)', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['BBM_20_2.0'],
                                mode='lines',
                                name='Middle BB',
                                line=dict(color='rgba(0,0,255,0.3)', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['BBL_20_2.0'],
                                mode='lines',
                                name='Lower BB',
                                line=dict(color='rgba(0,0,255,0.5)', width=1),
                                fill='tonexty',
                                fillcolor='rgba(0,0,255,0.05)'
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{selected_stock} Technical Analysis",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            font=dict(family="Comic Sans MS, cursive, sans-serif"),
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0.03)',
                            margin=dict(t=80, b=50, l=20, r=20)
                        )
                        
                        # Add grid lines
                        fig.update_xaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(0,0,0,0.1)'
                        )
                        
                        fig.update_yaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(0,0,0,0.1)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add secondary indicators in separate charts if selected
                        if "RSI" in tech_indicators:
                            fig_rsi = go.Figure()
                            
                            fig_rsi.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['RSI'],
                                mode='lines',
                                name='RSI',
                                line=dict(color='purple', width=2)
                            ))
                            
                            # Add overbought/oversold lines
                            fig_rsi.add_shape(
                                type="line",
                                x0=ta_df.index[0],
                                y0=70,
                                x1=ta_df.index[-1],
                                y1=70,
                                line=dict(color="red", width=1, dash="dash")
                            )
                            
                            fig_rsi.add_shape(
                                type="line",
                                x0=ta_df.index[0],
                                y0=30,
                                x1=ta_df.index[-1],
                                y1=30,
                                line=dict(color="green", width=1, dash="dash")
                            )
                            
                            fig_rsi.update_layout(
                                title="Relative Strength Index (RSI)",
                                xaxis_title="Date",
                                yaxis_title="RSI",
                                font=dict(family="Comic Sans MS, cursive, sans-serif"),
                                hovermode="x unified",
                                yaxis=dict(range=[0, 100]),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0.03)',
                                margin=dict(t=50, b=50, l=20, r=20)
                            )
                            
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        if "MACD" in tech_indicators and 'MACD_12_26_9' in ta_df.columns:
                            fig_macd = go.Figure()
                            
                            fig_macd.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['MACD_12_26_9'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig_macd.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['MACDs_12_26_9'],
                                mode='lines',
                                name='Signal',
                                line=dict(color='red', width=1)
                            ))
                            
                            # Add MACD histogram
                            colors = ['green' if val >= 0 else 'red' for val in ta_df['MACDh_12_26_9']]
                            
                            fig_macd.add_trace(go.Bar(
                                x=ta_df.index,
                                y=ta_df['MACDh_12_26_9'],
                                name='Histogram',
                                marker_color=colors
                            ))
                            
                            fig_macd.update_layout(
                                title="Moving Average Convergence Divergence (MACD)",
                                xaxis_title="Date",
                                yaxis_title="MACD",
                                font=dict(family="Comic Sans MS, cursive, sans-serif"),
                                hovermode="x unified",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0.03)',
                                margin=dict(t=50, b=50, l=20, r=20)
                            )
                            
                            st.plotly_chart(fig_macd, use_container_width=True)
                        
                        if "Stochastic Oscillator" in tech_indicators and 'STOCHk_14_3_3' in ta_df.columns:
                            fig_stoch = go.Figure()
                            
                            fig_stoch.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['STOCHk_14_3_3'],
                                mode='lines',
                                name='%K',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig_stoch.add_trace(go.Scatter(
                                x=ta_df.index,
                                y=ta_df['STOCHd_14_3_3'],
                                mode='lines',
                                name='%D',
                                line=dict(color='red', width=1)
                            ))
                            
                            # Add overbought/oversold lines
                            fig_stoch.add_shape(
                                type="line",
                                x0=ta_df.index[0],
                                y0=80,
                                x1=ta_df.index[-1],
                                y1=80,
                                line=dict(color="red", width=1, dash="dash")
                            )
                            
                            fig_stoch.add_shape(
                                type="line",
                                x0=ta_df.index[0],
                                y0=20,
                                x1=ta_df.index[-1],
                                y1=20,
                                line=dict(color="green", width=1, dash="dash")
                            )
                            
                            fig_stoch.update_layout(
                                title="Stochastic Oscillator",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                font=dict(family="Comic Sans MS, cursive, sans-serif"),
                                hovermode="x unified",
                                yaxis=dict(range=[0, 100]),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0.03)',
                                margin=dict(t=50, b=50, l=20, r=20)
                            )
                            
                            st.plotly_chart(fig_stoch, use_container_width=True)
                        
                        # Price Prediction Section
                        st.markdown("<h3>Price Prediction</h3>", unsafe_allow_html=True)
                        
                        # Use Linear Regression for simple prediction
                        if len(ta_df) > 30:  # Need enough data for prediction
                            # Prepare data for prediction
                            df_pred = ta_df[['Close']].copy()
                            df_pred['Prediction'] = df_pred['Close'].shift(-prediction_days)
                            
                            # Create features (using last 30 days of data)
                            X = np.array(range(30))
                            X = X.reshape(-1, 1)
                            
                            # Get the last 30 days of the close price
                            y = df_pred['Close'].values[-30:]
                            
                            # Create and train the model
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            # Predict the next 'prediction_days' days
                            x_forecast = np.array(range(30, 30 + prediction_days))
                            x_forecast = x_forecast.reshape(-1, 1)
                            forecast = model.predict(x_forecast)
                            
                            # Create a date range for the forecast
                            last_date = ta_df.index[-1]
                            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
                            
                            # Create a DataFrame for the forecast
                            forecast_df = pd.DataFrame({
                                'Date': forecast_dates,
                                'Forecast': forecast
                            })
                            
                            # Create a prediction chart
                            fig_pred = go.Figure()
                            
                            # Add actual price
                            fig_pred.add_trace(go.Scatter(
                                x=ta_df.index[-60:],  # Show last 60 days
                                y=ta_df['Close'][-60:],
                                mode='lines',
                                name='Actual Price',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Add prediction
                            fig_pred.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=forecast_df['Forecast'],
                                mode='lines',
                                name='Predicted Price',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            # Add confidence interval (simple approach)
                            std_dev = np.std(ta_df['Close'][-30:])
                            upper_bound = forecast + 2 * std_dev
                            lower_bound = forecast - 2 * std_dev
                            
                            fig_pred.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=upper_bound,
                                mode='lines',
                                name='Upper Bound',
                                line=dict(color='rgba(255,0,0,0.2)', width=0),
                                showlegend=False
                            ))
                            
                            fig_pred.add_trace(go.Scatter(
                                x=forecast_df['Date'],
                                y=lower_bound,
                                mode='lines',
                                name='Lower Bound',
                                line=dict(color='rgba(255,0,0,0.2)', width=0),
                                fill='tonexty',
                                fillcolor='rgba(255,0,0,0.1)',
                                showlegend=False
                            ))
                            
                            # Update layout
                            fig_pred.update_layout(
                                title=f"{selected_stock} Price Prediction (Next {prediction_days} Days)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                font=dict(family="Comic Sans MS, cursive, sans-serif"),
                                hovermode="x unified",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0.03)',
                                margin=dict(t=80, b=50, l=20, r=20)
                            )
                            
                            # Add grid lines
                            fig_pred.update_xaxes(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(0,0,0,0.1)'
                            )
                            
                            fig_pred.update_yaxes(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(0,0,0,0.1)'
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Display prediction summary
                            current_price = ta_df['Close'].iloc[-1]
                            predicted_price = forecast[-1]
                            change = ((predicted_price - current_price) / current_price) * 100
                            
                            pred_color = "green" if change >= 0 else "red"
                            
                            st.markdown(f"""
                            <div style='background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: 20px;'>
                                <h3 style='margin: 0; color: #2A2A72;'>Prediction Summary</h3>
                                <p style='margin: 10px 0;'>Current Price: <b>${current_price:.2f}</b></p>
                                <p style='margin: 10px 0;'>Predicted Price (in {prediction_days} days): <b style='color: {pred_color};'>${predicted_price:.2f}</b></p>
                                <p style='margin: 10px 0;'>Expected Change: <b style='color: {pred_color};'>{change:.2f}%</b></p>
                                <p style='font-size: 12px; color: gray; margin-top: 20px;'>Note: This is a simple linear prediction and should not be used as financial advice.</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No portfolio data available. Please upload a portfolio PDF.")
        
        # Individual Portfolio Tabs
        for i, portfolio_name in enumerate(st.session_state.portfolios.keys(), 1):
            with selected_tab[i]:
                df = st.session_state.portfolios[portfolio_name].copy()
                
                # Get current prices
                current_prices = get_stock_prices(df['Stock'].unique())
                
                # Calculate portfolio metrics
                df['Current Price'] = df['Stock'].map(current_prices)
                df['Current Value'] = df['Shares'] * df['Current Price']
                df['Purchase Value'] = df['Shares'] * df['Purchase Price']
                df['Gain/Loss'] = df['Current Value'] - df['Purchase Value']
                df['Gain %'] = (df['Current Value'] - df['Purchase Value']) / df['Purchase Value'] * 100
                
                # Display portfolio metrics
                col1, col2, col3 = st.columns(3)
                
                total_value = df['Current Value'].sum()
                total_cost = df['Purchase Value'].sum()
                total_gain = total_value - total_cost
                total_gain_pct = (total_gain / total_cost) * 100 if total_cost > 0 else 0
                
                gain_color = "green" if total_gain >= 0 else "red"
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                        <h3 style='margin: 0; color: #2A2A72;'>Total Value</h3>
                        <p style='font-size: 24px; font-weight: bold; margin: 10px 0;'>${total_value:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                        <h3 style='margin: 0; color: #2A2A72;'>Total Gain/Loss</h3>
                        <p style='font-size: 24px; font-weight: bold; margin: 10px 0; color: {gain_color};'>${total_gain:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                        <h3 style='margin: 0; color: #2A2A72;'>Return</h3>
                        <p style='font-size: 24px; font-weight: bold; margin: 10px 0; color: {gain_color};'>{total_gain_pct:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display the portfolio table
                st.markdown("<h3>Holdings</h3>", unsafe_allow_html=True)
                
                # Format the dataframe for display
                display_df = df.copy()
                display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
                display_df['Purchase Price'] = display_df['Purchase Price'].map('${:,.2f}'.format)
                display_df['Current Value'] = display_df['Current Value'].map('${:,.2f}'.format)
                display_df['Purchase Value'] = display_df['Purchase Value'].map('${:,.2f}'.format)
                display_df['Gain/Loss'] = display_df['Gain/Loss'].map('${:,.2f}'.format)
                display_df['Gain %'] = display_df['Gain %'].map('{:,.2f}%'.format)
                
                st.dataframe(
                    display_df,
                    column_config={
                        "Stock": st.column_config.TextColumn("Stock Symbol"),
                        "Shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                        "Purchase Price": st.column_config.TextColumn("Purchase Price"),
                        "Current Price": st.column_config.TextColumn("Current Price"),
                        "Purchase Value": st.column_config.TextColumn("Purchase Value"),
                        "Current Value": st.column_config.TextColumn("Current Value"),
                        "Gain/Loss": st.column_config.TextColumn("Gain/Loss"),
                        "Gain %": st.column_config.TextColumn("Return %"),
                        "Term": st.column_config.TextColumn("Term")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Portfolio Allocation Chart
                st.markdown("<h3>Portfolio Allocation</h3>", unsafe_allow_html=True)
                
                fig = px.pie(
                    df, 
                    values='Current Value', 
                    names='Stock',
                    title=f'{portfolio_name} Allocation by Value',
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    hole=0.4
                )
                
                fig.update_layout(
                    font=dict(family="Comic Sans MS, cursive, sans-serif", size=14),
                    title_font=dict(family="Comic Sans MS, cursive, sans-serif", size=20),
                    legend_title_font=dict(family="Comic Sans MS, cursive, sans-serif"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=50, b=50, l=20, r=20)
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#FFFFFF', width=2))
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Display welcome message and instructions when no portfolio is uploaded
        st.markdown("""
        <div style='background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: 50px;' class='floating'>
            <h2 style='color: #2A2A72; font-family: "Comic Sans MS", cursive, sans-serif;'>Welcome to DAJANII Portfolio Analyzer!</h2>
            <p style='font-size: 18px; margin: 20px 0;'>Upload your portfolio PDF to get started with advanced analysis and visualization.</p>
            <div style='margin-top: 30px;'>
                <img src='data:image/png;base64,{logo_base64}' width='150'>
            </div>
            <p style='font-size: 14px; color: gray; margin-top: 30px;'>Our tool helps you analyze your investments, track performance, and make informed decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features showcase
        st.markdown("""
        <div style='margin-top: 50px;'>
            <h2 style='color: #FF6B6B; font-family: "Comic Sans MS", cursive, sans-serif;'>✨ Features</h2>
            
            <div style='display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px;'>
                <div style='flex: 1; min-width: 250px; background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h3 style='color: #4ECDC4; font-family: "Comic Sans MS", cursive, sans-serif;'>📊 Portfolio Analysis</h3>
                    <p>Track your investments, analyze performance, and visualize your portfolio allocation.</p>
                </div>
                
                <div style='flex: 1; min-width: 250px; background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h3 style='color: #4ECDC4; font-family: "Comic Sans MS", cursive, sans-serif;'>📈 Technical Analysis</h3>
                    <p>Access advanced technical indicators like RSI, MACD, Bollinger Bands, and more.</p>
                </div>
                
                <div style='flex: 1; min-width: 250px; background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h3 style='color: #4ECDC4; font-family: "Comic Sans MS", cursive, sans-serif;'>🔮 Price Prediction</h3>
                    <p>Get price forecasts based on historical data and technical analysis.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='margin-top: 50px; text-align: center; padding: 20px; color: #666;'>
    <p>DAJANII Portfolio Analyzer © 2025 | Powered by AI</p>
    <p style='font-size: 12px;'>Disclaimer: This tool is for informational purposes only and should not be considered financial advice.</p>
</div>
""", unsafe_allow_html=True)
