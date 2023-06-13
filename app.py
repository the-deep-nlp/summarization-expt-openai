import os
import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

API_KEY = os.environ.get("API_KEY", None)

openai.api_key = API_KEY

original_text = st.text_area("Enter the original text")
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)

summarize_btn = st.button("Summarize")
if summarize_btn:
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Please summarize the following sentences: " + original_text,
            max_tokens=512,
            temperature=temperature
        )
    except Exception as exc:
        st.error(f"Some error occurred. Wrong API Key or other errors: {str(exc)}")
    else:
        summarized_text = response["choices"][0]["text"]
        if summarized_text:
            st.subheader("Summary")
            st.info(summarized_text)