# DAJANIII Stock Trading Assistant

**DAJANIII** is an advanced AI-powered stock trading assistant built with Streamlit. It provides tax-aware portfolio analytics, options-based hedging suggestions, technical analysis, and real-time market and crypto data.

## 🔧 Features
- Portfolio upload (CSV for Schwab, PDF for Interactive Brokers)
- Live market index and crypto data with sentiment analysis
- Tax-aware gain/loss calculation and estimated tax impact
- Options-based hedging strategies
- Technical indicators (RSI, MACD, MA, Bollinger Bands)
- News and political updates
- AI chat assistant powered by OpenAI

## 🚀 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/dajaniii-app.git
cd dajaniii-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

## 📦 Deployment

### Streamlit Cloud:
- Upload this repo to GitHub
- Go to https://streamlit.io/cloud and connect your GitHub
- Set main file as `app.py` and add your OpenAI API key under Secrets:
  ```
  OPENAI_API_KEY = "sk-..."
  ```

## 🐳 Docker (Optional)
```dockerfile
FROM python:3.11

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
```
