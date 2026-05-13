import streamlit as st
from memory_extractor import simple_extract
from graph_retriever import MemoryGraph

st.set_page_config(page_title="Knowledge Graph Memory", page_icon="🕸️")
st.title("Knowledge Graph Memory Demo")

if "memory_graph" not in st.session_state:
    st.session_state.memory_graph = MemoryGraph()

text = st.text_area("Enter memory text", "Mayukh studies LLM evaluation. Mayukh builds RAG systems.")

if st.button("Extract and add"):
    triples = simple_extract(text)
    for t in triples:
        st.session_state.memory_graph.add_triple(t.subject, t.relation, t.object)
    st.success(f"Added {len(triples)} triples")

entity = st.text_input("Search entity", "Mayukh")

if st.button("Search"):
    results = st.session_state.memory_graph.search_entity(entity)
    st.write(results)
