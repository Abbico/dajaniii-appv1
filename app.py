
import streamlit as st
import requests

st.set_page_config(page_title="DAJANIII Assistant", layout="wide")

st.title("🤖 DAJANIII Stock Trading Assistant")

# Settings
st.sidebar.header("Settings")

use_openrouter = st.sidebar.radio("Choose AI Provider:", ["OpenRouter", "OpenAI"], index=0)

if use_openrouter == "OpenRouter":
    openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")
else:
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

query = st.text_input("Ask a question about your stock portfolio...")

if query:
    if use_openrouter == "OpenRouter":
        if not openrouter_key:
            st.warning("Please enter your OpenRouter API Key.")
        else:
            with st.spinner("Thinking..."):
                headers = {
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful financial assistant."},
                        {"role": "user", "content": query}
                    ]
                }
                try:
                    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                    response.raise_for_status()
                    content = response.json()['choices'][0]['message']['content']
                    st.markdown(content)
                except Exception as e:
                    st.error(f"OpenRouter error: {str(e)}")
    else:
        st.warning("OpenAI support is not enabled yet. Add your key to use it.")
