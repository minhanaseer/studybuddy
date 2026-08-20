# Study Buddy

An AI-powered study assistant that answers questions from your own lecture notes/PDFs, and falls back to researching the web when the answer isn't in your materials.

## Status
🚧 Work in progress — currently in initial setup phase.

## Tech Stack
- **Language:** Python
- **UI:** Streamlit
- **PDF extraction:** PyMuPDF
- **Embeddings:** sentence-transformers (local, free)
- **Vector store:** FAISS
- **LLM:** TBD (planned: Ollama for local inference)

## Setup

### Prerequisites
- Python 3 installed
- macOS/Linux terminal (or equivalent)

### Installation

1. **Clone the repo and enter the project folder**
```bash
   git clone <your-repo-url>
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

## Roadmap
- [x] Initial Streamlit setup
- [ ] PDF upload and text extraction
- [ ] Chunking and embedding pipeline
- [ ] Vector search (FAISS) for Q&A over uploaded documents
- [ ] Web search fallback for questions not covered in notes
- [ ] Source citations in answers
- [ ] Dockerize the app
- [ ] Deploy live demo

## What I Learned
*(Fill this in as you build — interviewers love seeing this section. Note any design decisions, trade-offs, or bugs you had to solve.)*
