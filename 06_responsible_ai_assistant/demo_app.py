import streamlit as st
from safety_classifier import classify_and_rewrite

st.set_page_config(page_title="Responsible AI Assistant", page_icon="🛡️")
st.title("Responsible AI Assistant Demo")

text = st.text_area("Paste a model response to classify and rewrite safely")

if st.button("Analyze"):
    result = classify_and_rewrite(text)
    st.subheader("Result")
    st.json(result.__dict__)
