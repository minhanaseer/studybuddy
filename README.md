# Study Buddy

An AI-powered study assistant that answers questions from your own lecture notes/PDFs, and (planned) falls back to researching the web when the answer isn't in your materials.

## Status
🚧 Work in progress — PDF upload, extraction, chunking, embeddings, and similarity search are all working end-to-end. LLM answer generation not yet built.

## Tech Stack
- **Language:** Python
- **UI:** Streamlit
- **PDF extraction:** PyMuPDF
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`, local, free)
- **Vector store:** FAISS
- **LLM:** TBD (planned: Ollama for local inference)

## Setup

### Prerequisites
- Python 3 installed
- macOS/Linux terminal (or equivalent)

### Installation

1. **Clone the repo and enter the project folder**
```bash
   git clone https://github.com/YOUR_USERNAME/studybuddy.git
   cd studybuddy
```

2. **Create a virtual environment**
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install streamlit pymupdf sentence-transformers faiss-cpu
```

4. **Run the app**
```bash
   streamlit run app.py
```

   This opens a browser tab at `localhost:8501`.

## How it works so far
1. Upload a PDF through the file uploader
2. Text is extracted from every page using PyMuPDF
3. The text is split into overlapping chunks (200 words, 30-word overlap)
4. Each chunk is converted into an embedding vector using a local sentence-transformers model
5. Embeddings are stored in a FAISS index
6. When the user types a question, it's embedded the same way, and FAISS returns the 3 most similar chunks based on L2 (Euclidean) distance between vectors

## Roadmap
- [x] Initial Streamlit setup
- [x] PDF upload and text extraction
- [x] Chunking pipeline
- [x] Embedding pipeline (sentence-transformers)
- [x] FAISS vector store setup
- [x] Similarity search: retrieve relevant chunks for a user question
- [ ] LLM integration to generate grounded answers (instead of showing raw chunks)
- [ ] Confidence threshold: detect when no chunk is a strong match
- [ ] Web search fallback for questions not covered in notes
- [ ] Source citations in answers
- [ ] Dockerize the app
- [ ] Deploy live demo

## What I Learned
- Chunk size matters a lot depending on document type — 500-word chunks were too coarse for a short slide-deck PDF (only ~2 chunks total). Switched to 200-word chunks with 30-word overlap for finer-grained retrieval.
- Similarity search alone isn't enough for vague questions (e.g. "summarize this document") — it always returns the *k* closest chunks even if none are actually relevant. The L2 distance score is a useful signal for this: closely-bunched distances across all top matches (e.g. 1.89, 1.90, 2.01) indicate none of the results are a strong match, whereas a clear gap between the top result and the rest indicates high confidence.
- PyMuPDF's `fitz` import name is being deprecated in favor of `import pymupdf` directly.

## Architecture
