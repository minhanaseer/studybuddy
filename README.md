# Study Buddy

An AI-powered study assistant that answers questions from your own lecture notes/PDFs, grounded in the actual document content using RAG (Retrieval-Augmented Generation). Planned: fall back to web research when the answer isn't in your materials.

## Status
✅ Core RAG pipeline is fully working end-to-end: upload a PDF, ask a question, get an AI-generated answer grounded in your document, with sources shown.

## Tech Stack
- **Language:** Python
- **UI:** Streamlit
- **PDF extraction:** PyMuPDF
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`, local, free)
- **Vector store:** FAISS
- **LLM:** Ollama (running `llama3.2` locally, free, no API costs)

## Setup

### Prerequisites
- Python 3 installed
- [Ollama](https://ollama.com) installed (`brew install ollama`)
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
   pip install streamlit pymupdf sentence-transformers faiss-cpu ollama
```

4. **Pull the LLM model and start Ollama** (in a separate terminal tab, leave running)
```bash
   ollama pull llama3.2
   ollama serve
```

5. **Run the app**
```bash
   streamlit run app.py
```

   This opens a browser tab at `localhost:8501`.

## How it works
1. Upload a PDF through the file uploader
2. Text is extracted from every page using PyMuPDF
3. The text is split into overlapping chunks (200 words, 30-word overlap)
4. Each chunk is converted into an embedding vector using a local sentence-transformers model
5. Embeddings are stored in a FAISS index
6. When the user asks a question, it's embedded the same way, and FAISS retrieves the 3 most similar chunks
7. Those chunks are passed as context to a local LLM (Llama 3.2 via Ollama), which generates a grounded answer — instructed to say "I couldn't find this in your notes" if the context doesn't contain the answer
8. Source chunks used are shown in a collapsible section for verification

## Roadmap
- [x] PDF upload and text extraction
- [x] Chunking pipeline
- [x] Embedding pipeline (sentence-transformers)
- [x] FAISS vector store + similarity search
- [x] LLM integration for grounded answer generation (Ollama + Llama 3.2)
- [ ] Confidence threshold: detect low-relevance retrieval and warn the user
- [ ] Web search fallback for questions not covered in notes
- [ ] Inline source citations (page numbers, not just raw chunk text)
- [ ] Support multiple PDFs at once
- [ ] Dockerize the app
- [ ] Deploy live demo

## What I Learned
- Chunk size matters a lot depending on document type — 500-word chunks were too coarse for a short slide-deck PDF. Switched to 200-word chunks with 30-word overlap for finer-grained retrieval.
- Similarity search alone isn't enough for vague questions (e.g. "summarize this document") — it always returns the *k* closest chunks even if none are truly relevant. The L2 distance score is a useful confidence signal: closely-bunched distances across all top matches suggest a weak match overall.
- Prompt design matters for grounding: explicitly instructing the LLM to only use the provided context (and to say when it can't find an answer) reduces hallucination compared to a plain question-answering prompt.
- Running the LLM locally via Ollama avoids API costs entirely, but requires the `ollama serve` background process to stay running — a good reminder that local-first tools trade convenience for control.

## Architecture
