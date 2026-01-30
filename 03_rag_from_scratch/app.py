import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from chunking import chunk_text
from embeddings import EmbeddingModel
from retrieval import SimpleVectorIndex, DocumentChunk
from answer_with_citations import answer_question


load_dotenv()

st.set_page_config(page_title="PDF RAG Demo", page_icon="📄")
st.title("Citation-Grounded PDF RAG Demo")

api_key = st.sidebar.text_input("OpenAI API key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question about the PDF")

@st.cache_resource
def get_embedder():
    return EmbeddingModel()

def read_pdf(file_path: str):
    reader = PdfReader(file_path)
    chunks = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in chunk_text(text):
            chunks.append(DocumentChunk(text=chunk, source="uploaded_pdf", page=page_idx))
    return chunks

if uploaded and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Indexing PDF..."):
        chunks = read_pdf(tmp_path)
        embedder = get_embedder()
        vectors = embedder.encode([c.text for c in chunks])
        index = SimpleVectorIndex()
        index.add(chunks, vectors)

    with st.spinner("Retrieving and answering..."):
        q_vec = embedder.encode([question])[0]
        results = index.search(q_vec, top_k=5)
        answer = answer_question(question, results)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved context")
    for chunk, score in results:
        st.markdown(f"**Page {chunk.page} — score {score:.3f}**")
        st.write(chunk.text[:900] + ("..." if len(chunk.text) > 900 else ""))
