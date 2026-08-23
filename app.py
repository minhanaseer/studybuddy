import streamlit as st
import pymupdf as fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

st.title('Study Buddy')

CONFIDENCE_THRESHOLD = 1.0  # tune this based on testing - lower distance = more similar

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

def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I couldn't find this in your notes."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

uploaded_file = st.file_uploader("Upload a lecture PDF", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    st.success(f"Extracted text from {len(doc)} pages ({len(full_text.split())} words)")

    chunks = chunk_text(full_text)
    st.success(f"Split into {len(chunks)} chunks")

    with st.spinner("Creating embeddings..."):
        embeddings = model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.session_state['chunks'] = chunks
        st.session_state['index'] = index

    st.success(f"Created {len(embeddings)} embeddings")

    st.divider()
    st.subheader("Ask a question about this document")
    question = st.text_input("Your question")

    if question:
        question_embedding = model.encode([question]).astype('float32')
        k = 3
        distances, indices = st.session_state['index'].search(question_embedding, k)
        retrieved_chunks = [st.session_state['chunks'][idx] for idx in indices[0]]

        top_distance = distances[0][0]

        # Confidence check: is the best match actually close enough to trust?
        if top_distance > CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Low confidence match (distance: {top_distance:.2f}). "
                "This question might not be well-covered in your notes. "
                "The answer below may be unreliable."
            )
        else:
            st.info(f"✅ Good match found (distance: {top_distance:.2f})")

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, retrieved_chunks)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("See source chunks used"):
            for rank, idx in enumerate(indices[0]):
                st.markdown(f"**Source {rank+1}** (distance: {distances[0][rank]:.2f})")
                st.text(st.session_state['chunks'][idx])
