import streamlit as st
import pymupdf as fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

st.set_page_config(page_title="Study Buddy", page_icon="◆", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background-color: #0D0D0D; color: #E8E8E8; }
h1 { font-weight: 700; letter-spacing: -0.02em; color: #F2F2F2; border-bottom: 1px solid #2A2A2A; padding-bottom: 0.6rem; }
h2, h3 { color: #F2F2F2; font-weight: 600; }
[data-testid="stFileUploader"] { background-color: #161616; border: 1px dashed #3A3A3A; border-radius: 8px; padding: 1rem; }
.stTextInput input { background-color: #161616; color: #E8E8E8; border: 1px solid #2A2A2A; border-radius: 6px; }
.stTextInput input:focus { border-color: #5B8DEE; box-shadow: 0 0 0 1px #5B8DEE; }
div[data-testid="stAlert"] { background-color: #161616; border: 1px solid #2A2A2A; border-radius: 6px; color: #C9C9C9; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; }
hr { border-color: #2A2A2A; }
.streamlit-expanderHeader { background-color: #161616; border: 1px solid #2A2A2A; border-radius: 6px; color: #C9C9C9; }
.stTextArea textarea { background-color: #131313; color: #A0A0A0; border: 1px solid #2A2A2A; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
.answer-block { background-color: #141820; border-left: 3px solid #5B8DEE; border-radius: 4px; padding: 1.2rem; margin-top: 0.5rem; line-height: 1.6; }
.not-found-block { background-color: #1F1812; border-left: 3px solid #D9A441; border-radius: 4px; padding: 1.2rem; margin-top: 0.5rem; line-height: 1.6; color: #D9C6A0; }
.stSpinner > div { color: #8A8A8A; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Study Buddy")
st.caption("Ask questions about your own lecture notes — answered only from what's actually in them.")

CONFIDENCE_THRESHOLD = 1.0

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def generate_answer_from_notes(question, retrieved_chunks_with_sources):
    context_parts = [f"[From {src}]\n{text}" for text, src in retrieved_chunks_with_sources]
    context = "\n\n".join(context_parts)
    prompt = f"""You are answering strictly based on provided context. Do not use any outside knowledge. Do not guess, infer, or fill in gaps with assumptions.

Rules:
1. Only state facts that are explicitly written in the context below.
2. If the context does not clearly contain the answer, respond exactly: "I couldn't find this in your notes."
3. Do not speculate about what the document "probably" says. Quote or closely paraphrase only what is actually there.

Context:
{context}

Question: {question}

Answer (strictly from the context above):"""
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1}
    )
    return response['message']['content']

uploaded_files = st.file_uploader("Upload lecture PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    all_sources = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        chunks = chunk_text(full_text)
        all_chunks.extend(chunks)
        all_sources.extend([uploaded_file.name] * len(chunks))
        st.success(f"{uploaded_file.name} — {len(doc)} pages, {len(chunks)} chunks")

    with st.spinner(f"Creating embeddings for {len(all_chunks)} chunks..."):
        embeddings = model.encode(all_chunks)
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        st.session_state['chunks'] = all_chunks
        st.session_state['sources'] = all_sources
        st.session_state['index'] = index

    st.success(f"Ready — {len(all_chunks)} chunks indexed across {len(uploaded_files)} document(s)")

    st.divider()
    st.subheader("Ask a question")
    question = st.text_input("", placeholder="e.g. How do I convert binary to decimal?")

    if question:
        question_embedding = model.encode([question]).astype('float32')
        k = min(3, len(st.session_state['chunks']))
        distances, indices = st.session_state['index'].search(question_embedding, k)
        top_distance = distances[0][0]

        if top_distance > CONFIDENCE_THRESHOLD:
            st.markdown(
                '<div class="not-found-block">'
                '<strong>⚠ Not found in the document(s) attached.</strong><br><br>'
                'This question doesn\'t appear to be covered in your uploaded notes. '
                'Try rephrasing, or check that the right document is uploaded.'
                '</div>',
                unsafe_allow_html=True
            )
            with st.expander("See closest chunks found anyway (for reference)"):
                for rank, idx in enumerate(indices[0]):
                    source_name = st.session_state['sources'][idx]
                    st.markdown(f"**{rank+1}.** *{source_name}* (distance: {distances[0][rank]:.2f})")
                    st.text(st.session_state['chunks'][idx])
        else:
            retrieved_chunks_with_sources = [
                (st.session_state['chunks'][idx], st.session_state['sources'][idx])
                for idx in indices[0]
            ]
            st.info(f"✓ Good match in your notes (distance: {top_distance:.2f})")
            with st.spinner("Generating answer..."):
                answer = generate_answer_from_notes(question, retrieved_chunks_with_sources)
            st.markdown("### Answer")
            st.markdown('<div class="answer-block">' + answer + '</div>', unsafe_allow_html=True)
            with st.expander("See source chunks used"):
                for rank, idx in enumerate(indices[0]):
                    source_name = st.session_state['sources'][idx]
                    st.markdown(f"**Source {rank+1}** — *{source_name}* (distance: {distances[0][rank]:.2f})")
                    st.text(st.session_state['chunks'][idx])
