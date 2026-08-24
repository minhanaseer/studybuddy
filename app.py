import streamlit as st
import pymupdf as fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

st.title('Study Buddy')

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

def generate_answer(question, retrieved_chunks_with_sources):
    context_parts = [f"[From {src}]\n{text}" for text, src in retrieved_chunks_with_sources]
    context = "\n\n".join(context_parts)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I couldn't find this in your notes." Mention which document the answer came from if relevant.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

uploaded_files = st.file_uploader("Upload lecture PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []       # will hold every chunk from every PDF
    all_sources = []      # parallel list: which filename each chunk came from

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        full_text = ""
        for page in doc:
            full_text += page.get_text()

        chunks = chunk_text(full_text)
        all_chunks.extend(chunks)
        all_sources.extend([uploaded_file.name] * len(chunks))

        st.success(f"{uploaded_file.name}: extracted {len(doc)} pages, split into {len(chunks)} chunks")

    with st.spinner(f"Creating embeddings for {len(all_chunks)} total chunks..."):
        embeddings = model.encode(all_chunks)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.session_state['chunks'] = all_chunks
        st.session_state['sources'] = all_sources
        st.session_state['index'] = index

    st.success(f"Ready: {len(all_chunks)} chunks indexed across {len(uploaded_files)} document(s)")

    st.divider()
    st.subheader("Ask a question about your documents")
    question = st.text_input("Your question")

    if question:
        question_embedding = model.encode([question]).astype('float32')
        k = 3
        distances, indices = st.session_state['index'].search(question_embedding, k)

        retrieved_chunks_with_sources = [
            (st.session_state['chunks'][idx], st.session_state['sources'][idx])
            for idx in indices[0]
        ]

        top_distance = distances[0][0]

        if top_distance > CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Low confidence match (distance: {top_distance:.2f}). "
                "This question might not be well-covered in your notes."
            )
        else:
            st.info(f"✅ Good match found (distance: {top_distance:.2f})")

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, retrieved_chunks_with_sources)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("See source chunks used"):
            for rank, idx in enumerate(indices[0]):
                source_name = st.session_state['sources'][idx]
                st.markdown(f"**Source {rank+1}** — *{source_name}* (distance: {distances[0][rank]:.2f})")
                st.text(st.session_state['chunks'][idx])
