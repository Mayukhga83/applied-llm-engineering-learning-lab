import os
import streamlit as st
from dotenv import load_dotenv

from toxicity_eval import heuristic_toxicity_eval
from bias_eval import heuristic_bias_eval

load_dotenv()

st.set_page_config(page_title="LLM Evaluation Demo", page_icon="🧪")
st.title("LLM Response Evaluation Demo")

api_key = st.sidebar.text_input("OpenAI API key, optional", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

question = st.text_area("User question", height=100)
answer = st.text_area("Model answer to evaluate", height=180)

if st.button("Evaluate"):
    tox = heuristic_toxicity_eval(answer)
    bias = heuristic_bias_eval(answer)

    st.subheader("Heuristic toxicity result")
    st.json(tox.__dict__)

    st.subheader("Heuristic bias result")
    st.json(bias.__dict__)

    st.info("Optional upgrade: use llm_judge.py with an OpenAI API key for rubric-based evaluation.")
