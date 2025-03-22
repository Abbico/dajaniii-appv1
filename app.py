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
