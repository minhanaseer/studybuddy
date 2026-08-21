# Study Buddy

An AI-powered study assistant that answers questions from your own lecture notes/PDFs, and (planned) falls back to researching the web when the answer isn't in your materials.

## Status
🚧 Work in progress — PDF upload, text extraction, chunking, and embeddings are working. Search/Q&A not yet built.

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

   This should open a browser tab at `localhost:8501` showing the app.

## How it works so far
1. Upload a PDF through the file uploader
2. Text is extracted from every page using PyMuPDF
3. The text is split into overlapping chunks (default: 500 words, 50-word overlap)
4. Each chunk is converted into an embedding vector using a local sentence-transformers model
5. Embeddings are stored in a FAISS index, ready for similarity search (coming next)

## Roadmap
- [x] Initial Streamlit setup
- [x] PDF upload and text extraction
- [x] Chunking pipeline
- [x] Embedding pipeline (sentence-transformers)
- [x] FAISS vector store setup
- [ ] Similarity search: retrieve relevant chunks for a user question
- [ ] LLM integration to generate grounded answers
- [ ] Web search fallback for questions not covered in notes
- [ ] Source citations in answers
- [ ] Dockerize the app
- [ ] Deploy live demo

## What I Learned
- Chunk size matters a lot depending on document type — a 500-word chunk size worked fine for dense text, but produced too few, too-coarse chunks on a short slide-deck PDF (only ~2 chunks total). Smaller chunk sizes (150–200 words) are likely better for slide-style content.
- PyMuPDF's `fitz` import name is being deprecated in favor of `import pymupdf` directly.
