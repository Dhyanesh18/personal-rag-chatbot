import streamlit as st
import time

def type_response(response, delay=0.02):
    placeholder = st.empty()
    displayed_text = ""
    for char in response:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(delay)