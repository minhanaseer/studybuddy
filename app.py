import streamlit as st
import pymupdf as fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title('Study Buddy')

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

uploaded_file = st.file_uploader("Upload a lecture PDF", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    st.success(f"Extracted text from {len(doc)} pages")

    chunks = chunk_text(full_text)
    st.success(f"Split into {len(chunks)} chunks")

    with st.spinner("Creating embeddings..."):
        embeddings = model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')

        # Build a FAISS index and add our embeddings to it
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save to session state so we can use it for search later
        st.session_state['chunks'] = chunks
        st.session_state['index'] = index

    st.success(f"Created {len(embeddings)} embeddings, each with {embeddings.shape[1]} dimensions")

    st.subheader("Preview chunks")
    chunk_index = st.selectbox("Pick a chunk to view", range(len(chunks)))
    st.text_area(f"Chunk {chunk_index}", chunks[chunk_index], height=250)
