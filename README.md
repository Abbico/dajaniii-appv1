# DAJANII Portfolio Analyzer

A fun, cartoonish portfolio analysis application that allows you to upload portfolio PDFs, analyze your investments, and get technical insights with advanced visualization tools.

## Features

- **Single PDF Upload**: Upload any portfolio PDF to extract holdings data
- **Combined Portfolio View**: Automatically combines all uploaded portfolios
- **Technical Analysis Tools**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator
- **Price Prediction**: Linear regression-based price forecasting with confidence intervals
- **Index Comparison**: Compare your portfolio performance with major market indexes
- **Fun, Cartoonish UI**: Vibrant colors, animations, and user-friendly interface

## Installation

1. Clone this repository or extract the zip file
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Launch the application using the command above
2. Upload any portfolio PDF file using the upload area
3. The app will extract holdings data and create a new portfolio
4. Each upload creates a separate portfolio and updates the combined view
5. Use the tabs to switch between individual portfolios and the combined view
6. Explore technical analysis tools and charts for your stocks
7. Compare performance with major market indexes
8. Get price predictions for your holdings

## PDF Format Support

The application supports various portfolio PDF formats and attempts to extract:
- Stock symbols (1-5 uppercase letters)
- Number of shares
- Purchase prices

Multiple pattern recognition algorithms are used to handle different portfolio formats.

## Technical Analysis Tools

- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **RSI**: 14-period Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: 20-period with 2 standard deviations
- **Stochastic Oscillator**: 14-period with 3-period smoothing

## Customization

You can customize the analysis settings in the sidebar:
- Time period for analysis
- Comparison indexes
- Technical indicators to display
- Prediction time horizon

## Requirements

See requirements.txt for a complete list of dependencies.

## Disclaimer

This tool is for informational purposes only and should not be considered financial advice.
