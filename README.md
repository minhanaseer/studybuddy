# Study Buddy

An AI-powered study assistant that answers questions from your own lecture notes/PDFs, grounded in the actual document content using RAG (Retrieval-Augmented Generation). Supports multiple documents at once, with source attribution. Planned: fall back to web research when the answer isn't in your materials.

## Status
✅ Core RAG pipeline is fully working end-to-end, with multi-document support: upload one or more PDFs, ask a question, get an AI-generated answer grounded in your documents, with per-document source attribution and a confidence indicator.

🚧 Next: web search fallback (Tavily API selected, integration pending) → Docker → live deployment.

## Tech Stack
- **Language:** Python
- **UI:** Streamlit
- **PDF extraction:** PyMuPDF
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`, local, free)
- **Vector store:** FAISS
- **LLM:** Ollama (running `llama3.2` locally, free, no API costs)
- **Web search (planned):** Tavily API

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
1. Upload one or more PDFs through the file uploader
2. Text is extracted from every page of every PDF using PyMuPDF
3. Each document's text is split into overlapping chunks (200 words, 30-word overlap), tagged with its source filename
4. All chunks (across all uploaded documents) are converted into embedding vectors using a local sentence-transformers model
5. Embeddings are stored in a single FAISS index, alongside a parallel list tracking each chunk's source document
6. When the user asks a question, it's embedded the same way, and FAISS retrieves the 3 most similar chunks across all documents
7. A confidence check compares the best match's distance against a threshold — if the match is weak, the user sees a warning that the answer may be unreliable
8. Retrieved chunks (tagged by source) are passed as context to a local LLM (Llama 3.2 via Ollama), which generates a grounded answer and can reference which document it came from
9. Source chunks used are shown in a collapsible section, labeled by filename, for verification

## Roadmap
- [x] PDF upload and text extraction
- [x] Chunking pipeline
- [x] Embedding pipeline (sentence-transformers)
- [x] FAISS vector store + similarity search
- [x] LLM integration for grounded answer generation (Ollama + Llama 3.2)
- [x] Confidence threshold to flag weak/irrelevant retrieval matches
- [x] Support multiple PDFs at once, with per-chunk source tracking
- [ ] Web search fallback for questions not covered in notes (Tavily API selected, integration pending)
- [ ] Dockerize the app
- [ ] Deploy live demo
- [ ] Record demo video/GIF

## What I Learned
- Chunk size matters a lot depending on document type — 500-word chunks were too coarse for a short slide-deck PDF. Switched to 200-word chunks with 30-word overlap for finer-grained retrieval.
- Similarity search alone isn't enough for vague questions (e.g. "summarize this document") — it always returns the *k* closest chunks even if none are truly relevant. Added a confidence threshold using the L2 distance score to flag weak matches explicitly to the user, rather than presenting every answer with equal confidence.
- Prompt design matters for grounding: explicitly instructing the LLM to only use the provided context (and to say when it can't find an answer) reduces hallucination compared to a plain question-answering prompt.
- Supporting multiple documents required tracking source alongside every chunk from the start (a parallel list, not just a flat list of text) — retrofitting source tracking after the fact would have been much messier than designing for it upfront.
- Running the LLM locally via Ollama avoids API costs entirely, but requires the `ollama serve` background process to stay running.

## Architecture

## Technical Deep Dive: How Chunking Actually Works

The chunking function uses **fixed-size, word-count-based chunking with overlap** — the simplest chunking strategy in RAG, chosen deliberately for its predictability and simplicity over more complex alternatives (e.g. sentence-based or semantic chunking).

```python
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
```

### How the sliding window works
Each new chunk starts `chunk_size - overlap` words after the previous chunk started (170 words forward, with the defaults above). This means each chunk shares its last 30 words with the start of the next chunk, preserving context across the cut point instead of splitting an idea cleanly in half.

**Worked example — an 830-word document, chunk_size=200, overlap=30:**

| Chunk # | start | end | Words used | Length |
|---|---|---|---|---|
| 1 | 0 | 200 | 0–200 | 200 |
| 2 | 170 | 370 | 170–370 | 200 |
| 3 | 340 | 540 | 340–540 | 200 |
| 4 | 510 | 710 | 510–710 | 200 |
| 5 | 680 | 830 | 680–830 | 150 (final chunk, ran out of words) |

Result: 5 chunks. The final chunk is shorter than the rest whenever the document length isn't an exact multiple of the step size — expected behavior, not a bug.

### General formula
This generalizes to any document length — a short PDF might produce 1 chunk, a full semester of combined notes might produce hundreds, using the exact same function with no changes needed.

### Why 200 words / 30-word overlap specifically
These values are **empirically chosen defaults, not derived from any formula**. The starting rule of thumb: overlap is typically 10–20% of chunk size (30/200 = 15% here). The chunk size itself was reduced from an initial 500-word default after testing showed it produced only ~2 chunks on a short slide-deck PDF — too coarse for precise retrieval. 200 words was chosen as a better fit for slide-style content, where each idea is already relatively short.

### Known limitation
This method cuts purely by word count, with no awareness of sentence or paragraph boundaries — a chunk can end mid-sentence. The word-overlap mitigates this somewhat (the cut-off idea also appears at the start of the next chunk), but doesn't fully solve it. More advanced approaches (sentence-boundary-aware chunking, or semantic chunking using the embedding model to detect topic shifts) could improve this, and are a possible future improvement.

### What "embedding dimension" means
Every chunk, regardless of its actual length, gets converted into a **fixed-size vector of 384 numbers** (using `all-MiniLM-L6-v2`). This dimension count is a property of the specific embedding model, not related to word/character count — a 5-word chunk and a 200-word chunk both produce exactly 384 numbers. Think of it as a fixed-length "fingerprint of meaning" for whatever text goes in.

### What FAISS actually stores
`IndexFlatL2` stores every chunk's embedding vector in memory (RAM) as-is, with no compression ("Flat"), and compares vectors using L2 (Euclidean) distance. On search, it does a brute-force comparison against every stored vector — exact and simple, appropriate at this project's scale (tens to low hundreds of chunks), though larger-scale RAG systems (millions of vectors) typically use approximate index types (e.g. `IndexIVFFlat`, `IndexHNSW`) that trade a little accuracy for much greater speed.

### On third-party components
This project orchestrates several external tools rather than building any of them from scratch: PyMuPDF (extraction), sentence-transformers (embeddings), FAISS (vector storage/search), and Ollama/Llama 3.2 (text generation). The engineering work is in the pipeline design, chunking/retrieval strategy, prompt engineering for grounding, and confidence handling — not in building the underlying models or libraries. This mirrors how most real-world "AI engineering" roles work: composing and orchestrating existing models and tools effectively, rather than training models from scratch.
