# Study Buddy

An AI-powered study assistant that answers questions strictly from your own uploaded lecture notes/PDFs, using RAG (Retrieval-Augmented Generation). If the answer isn't in your documents, it says so clearly instead of guessing or searching elsewhere.

## Status
✅ Complete, working RAG pipeline: upload one or more PDFs, ask a question, get an answer grounded strictly in your documents, with source attribution and a confidence check. No external knowledge, no web search — answers only from what you actually uploaded.

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
1. Upload one or more PDFs through the file uploader
2. Text is extracted from every page of every PDF using PyMuPDF
3. Each document's text is split into overlapping chunks (200 words, 30-word overlap), tagged with its source filename
4. All chunks (across all uploaded documents) are converted into embedding vectors using a local sentence-transformers model
5. Embeddings are stored in a single FAISS index, alongside a parallel list tracking each chunk's source document
6. When the user asks a question, it's embedded the same way, and FAISS retrieves the 3 most similar chunks across all documents
7. A confidence check compares the best match's distance against a threshold. If the match is weak, the app clearly states the answer isn't in the uploaded document(s) — no generation, no guessing, no external lookup
8. If the match is strong, retrieved chunks (tagged by source) are passed as context to a local LLM (Llama 3.2 via Ollama, temperature 0.1) with strict instructions to answer only from that context
9. Source chunks used are shown in a collapsible section, labeled by filename, for verification

## What this app can and can't answer
**Can answer:** specific, fact-based questions where the answer is explicitly stated somewhere in the uploaded document(s), across single or multiple files.

**Cannot answer:** anything not literally present in the uploaded documents (no external/web knowledge), vague meta-questions like "what is this about" or "summarize this" (these rarely match a specific retrieved chunk well), or questions requiring synthesis across an entire document rather than a specific passage.

## Roadmap
- [x] PDF upload and text extraction (multi-file)
- [x] Chunking pipeline
- [x] Embedding pipeline (sentence-transformers)
- [x] FAISS vector store + similarity search
- [x] LLM integration for grounded answer generation (Ollama + Llama 3.2, low temperature)
- [x] Confidence threshold that skips generation on weak matches
- [x] Multi-PDF support with per-chunk source tracking
- [x] Dark UI theme
- [ ] Dockerize the app
- [ ] Deploy live demo
- [ ] Record demo video/GIF

## What I Learned
- Chunk size matters a lot depending on document type — 500-word chunks were too coarse for a short slide-deck PDF. Switched to 200-word chunks with 30-word overlap for finer-grained retrieval.
- Similarity search alone isn't enough for vague questions (e.g. "summarize this document") — it always returns the *k* closest chunks even if none are truly relevant. A confidence threshold using the L2 distance score flags this, skipping generation entirely on weak matches rather than answering from irrelevant context.
- **Tried and reversed a web search fallback (Tavily API):** initially built a feature to search the web when a question wasn't covered in the notes. Testing revealed this was unreliable — vague or document-dependent questions (e.g. "what is the budget for this year") got sent to web search with no context about *which* budget, returning plausible-sounding but irrelevant answers (e.g. the US federal budget, when the actual document was something else entirely). Rather than build increasingly complex heuristics to detect "is this question really about my document," removed the web fallback entirely in favor of a simple, honest rule: if it's not clearly in the uploaded documents, say so. This was a genuine design reversal based on testing, not a plan followed blindly.
- Small local LLMs (like `llama3.2:3b`) are prone to filling gaps with plausible-sounding but fabricated content, especially at default temperature settings. Lowering temperature to 0.1 and writing explicit, numbered "do not guess" rules into the prompt meaningfully reduced this, though it doesn't fully eliminate the limitation of a small model.
- Supporting multiple documents required tracking source alongside every chunk from the start (a parallel list, not just a flat list of text) — designing for this upfront was far simpler than retrofitting it later.
- Running the LLM locally via Ollama avoids API costs entirely, but requires the `ollama serve` background process to stay running.

## Architecture

### Why 200 words / 30-word overlap specifically
Empirically chosen defaults, not derived from a formula. Overlap is typically 10–20% of chunk size (30/200 = 15% here). Chunk size was reduced from an initial 500-word default after testing showed it produced only ~2 chunks on a short slide-deck PDF — too coarse for precise retrieval.

### Known limitation
This method cuts purely by word count, with no awareness of sentence or paragraph boundaries — a chunk can end mid-sentence. More advanced approaches (sentence-boundary-aware chunking, or semantic chunking) could improve this.

### What "embedding dimension" means
Every chunk, regardless of length, becomes a **fixed-size vector of 384 numbers** (`all-MiniLM-L6-v2`). This is a property of the specific model, not related to word/character count.

### What FAISS actually stores
`IndexFlatL2` stores every chunk's embedding vector in memory (RAM), uncompressed, comparing vectors using L2 (Euclidean) distance via brute-force search — exact and simple, appropriate at this project's scale (tens to low hundreds of chunks).

### On temperature and grounding
LLM `temperature` controls output randomness. Default settings (~0.7-0.8) allow enough creative freedom that a small model can "fill gaps" with plausible-sounding but fabricated content when given weak or incomplete context. Setting `temperature=0.1` in the Ollama call, combined with explicit numbered rules in the prompt ("do not guess," "only state facts explicitly in the context"), meaningfully reduces this — though it's a mitigation, not a complete fix, given the limitations of a small (3B parameter) local model.

### On third-party components
This project orchestrates several external tools rather than building any of them from scratch: PyMuPDF (extraction), sentence-transformers (embeddings), FAISS (vector storage/search), and Ollama/Llama 3.2 (text generation). The engineering work is in the pipeline design, chunking/retrieval strategy, prompt engineering for grounding, confidence handling, and the decision to remove an unreliable feature (web search) in favor of a simpler, more honest one — not in building the underlying models or libraries.

## Experiment: Tuning Chunk Size for a Real-World Data Case

While testing multi-document retrieval with a spreadsheet-derived PDF (a catering budget converted from Excel), a specific retrieval failure surfaced: a question about a named caterer ("what is the item from Mr. Egg?") consistently failed the confidence threshold, even though the correct answer was present in the retrieved chunk.

**Root cause:** the source PDF's table structure was flattened into a single continuous text stream during extraction, so each 200-word chunk contained 10–15 unrelated caterers' data jumbled together. This diluted the embedding's ability to represent any single caterer distinctly.

**Chunk size was reduced in three steps to test whether smaller chunks would isolate the relevant row and improve the match:**

| Chunk size | "Mr Egg" question distance | Change |
|---|---|---|
| 200 words (original) | 1.49 | — |
| 60 words | 1.30 | -0.19 |
| 30 words | 1.27 | -0.03 |

**Finding: diminishing returns.** Cutting chunk size from 200→60 words gave a meaningful improvement (-0.19), but cutting further from 60→30 words gave almost none (-0.03), despite chunk size dropping by the same proportion. This indicates chunk-size tuning was approaching its ceiling for this failure mode, not scaling linearly with further reduction.

**Conclusion:** this points to a different underlying limitation than chunk size — semantic embedding models are built to capture *meaning*, and a specific proper noun like "Mr Egg" carries little distinct semantic content to the model (it doesn't "mean" much differently from "a caterer" in general). No amount of chunking adjustment fully resolves this, because the limitation sits in what the embedding model represents, not how the text is sliced.

**The correct fix for this class of problem (not yet implemented) would be hybrid search:** combining semantic embedding search with exact keyword matching, so specific names/terms are caught by literal text matching even when their semantic embedding is weak. This is a well-established pattern in production RAG systems for exactly this reason — embeddings and keyword search have complementary strengths.

**Practical takeaway:** chunk size tuning is a real, useful lever, but it has diminishing returns and doesn't fix every retrieval failure mode. Recognizing *when* a result plateaus (rather than continuing to shrink chunk size indefinitely) was itself a useful finding — it pointed to a different, more fundamental limitation worth understanding rather than chasing with the same tool.
