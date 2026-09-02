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

**Final decision:** reverted to `chunk_size=200, overlap=30` as the default. The app's primary use case is prose-heavy lecture notes, where larger chunks preserve complete explanations better than the fragmentation caused by very small chunks. The tabular-data retrieval issue documented above remains a known limitation for spreadsheet-style content specifically, not the general case.

## Dockerized Setup (Multi-Container)

The app is containerized using Docker Compose, running as two separate services rather than one monolithic container — the standard approach for applications with distinct components (an app server and a model-serving service) that have different resource needs and lifecycles.

### Architecture

### Files

**`requirements.txt`** — pinned Python dependencies (streamlit, pymupdf, sentence-transformers, faiss-cpu, ollama).

**`Dockerfile`** — builds the app's image:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
Dependencies are installed in a separate layer from the app code (`COPY requirements.txt` before `COPY app.py`) so Docker can cache the dependency-install step and skip re-running it when only the app code changes, not the dependencies — a standard Docker layer-caching optimization.

**`docker-compose.yml`** — orchestrates both containers:
```yaml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
```
The `ollama_data` named volume persists downloaded models across container restarts. `OLLAMA_HOST` uses `ollama` as a hostname — Docker Compose automatically resolves service names to internal container addresses, so the app container can reach the Ollama container without hardcoding an IP.

### Code change required

`app.py` originally called `ollama.chat()` directly, which implicitly connects to `localhost`. Inside Docker, "localhost" from the app container's perspective is the app container itself, not the Ollama container — so this had to change to an explicit client pointed at an environment variable:
```python
import os
ollama_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
```
This falls back to `localhost:11434` when the env var isn't set, so the same code works both locally (outside Docker) and inside the containerized setup without maintaining two versions.

### Running it

```bash
docker compose up --build
```
Builds the app image and starts both containers. First run downloads the Python base image, installs dependencies (`sentence-transformers`'s dependency on `torch` makes this step slow — took ~18 minutes on an 8GB RAM MacBook Air), and pulls the official Ollama image.

The Ollama container starts with no models downloaded. The model has to be pulled into the running container separately:
```bash
docker exec -it studybuddy-ollama-1 ollama pull llama3.2
```

### What I learned setting this up
- **PATH issues after installing Docker Desktop**: the `docker` CLI binary was installed but not linked into the shell's PATH, requiring a manual addition to `.zshrc` (`export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"`) and a shell restart — a common but non-obvious first-time setup snag.
- **Resource constraints on 8GB RAM hardware**: running Docker Desktop (which itself runs a Linux VM) alongside Ollama and a model is genuinely close to the practical RAM ceiling on a base MacBook Air. Build times were noticeably slower than typical, and this is a real, honest constraint of developing on modest hardware rather than a sign anything was configured wrong.
- **Disk usage**: the full setup (base image, Python dependencies including torch, Ollama image, and the pulled model) totals roughly 5-7GB. `docker system df` and `docker system prune -a` are useful for monitoring and reclaiming space, especially relevant with limited free disk space.
- **Multi-container vs. single-container tradeoff**: chose to run the app and Ollama as separate services rather than bundling everything into one image. This is more representative of real deployment patterns (separate services scale and update independently) even though it required understanding Docker Compose networking (service-name-based hostnames) rather than just a single Dockerfile.

## Experiment: Exact-Match Retrieval for Numbered References

Testing with a real legal document (a 17-page tenancy agreement, 50 chunks) surfaced another retrieval failure related to the earlier "Mr Egg" finding: a question about a specific numbered clause ("what does clause 6.4 say") returned "not found," even though the clause was genuinely present in the document.

**Root cause (same underlying issue as the proper-noun case):** legal documents contain dozens of similarly-structured numbered references (4.15.29, 9.1.1, 6.3.2, etc.). Embedding models capture semantic meaning, not exact numeric identifiers — "6.4" doesn't carry distinct meaning relative to "9.1" or "4.15" in vector space, so semantic search struggles to reliably distinguish between them, especially in a document with this many numbered clauses.

**Fix implemented:** rather than trying to force semantic search to handle this case (which has an inherent ceiling, as the earlier chunk-size experiment showed), added a targeted exact-match step that runs *before* semantic search:

```python
def find_clause_number(question):
    match = re.search(r'\b(?:clause|section)\s*(\d+(?:\.\d+)*)', question, re.IGNORECASE)
    if match:
        return match.group(1)
    bare_match = re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', question)
    if bare_match:
        return bare_match.group(1)
    return None

def find_chunks_containing_clause(chunks, sources, clause_number):
    pattern = re.compile(r'\b' + re.escape(clause_number) + r'\b')
    matches = []
    for i, chunk in enumerate(chunks):
        if pattern.search(chunk):
            matches.append((chunk, sources[i]))
    return matches
```

If the question contains a recognizable clause/section number pattern, the app searches chunk text directly for that exact string (with word boundaries, to avoid "6.4" matching inside "16.4" or "6.40") and skips vector search entirely for that query. Only falls back to normal FAISS similarity search when no clause number pattern is detected.

**Why this is the right fix rather than more chunk-size tuning:** exact identifiers (clause numbers, proper nouns, product codes) are fundamentally a poor fit for semantic search — no amount of chunking adjustment reliably solves it, since the limitation is in what embeddings represent. A targeted exact-match check for structured, identifiable patterns is a standard technique that complements semantic search rather than trying to make one approach do everything.

**Generalization:** this same pattern (detect a structured identifier in the question, exact-match search before falling back to embeddings) could extend to other identifiable patterns beyond clause numbers — e.g. dates, reference codes, or specific named entities — as a lighter-weight alternative to full hybrid search for a bounded set of known pattern types.

## Design Note: Two Independent Grounding Checks

A question can show "Good match" (passing the retrieval confidence check) and still return "I couldn't find this in your notes" — this is expected behavior, not a bug, and reflects two deliberately independent safeguards:

1. **Retrieval confidence (distance threshold)** — answers "is the closest chunk even topically relevant?" A low distance means FAISS found a chunk in the right neighborhood, but says nothing about whether that chunk contains the *specific* fact being asked about.

2. **Generation strictness (the prompt's grounding rules)** — once a topically-relevant chunk is retrieved, the LLM independently judges whether that chunk actually, clearly states the answer to the specific question asked. The prompt explicitly instructs it to say "I couldn't find this in your notes" rather than paraphrase loosely or guess, even from relevant-but-not-quite-on-point context.

**Both checks have to pass for a real answer to be generated.** A chunk can be topically close (e.g., a chunk about tenancy termination, when asked "what's the idea of leaving without informing") without directly answering the specific phrasing of the question — in which case retrieval succeeds but generation correctly declines. This two-layer approach trades away some "helpfulness" (it won't stretch a loosely-related chunk into a confident-sounding answer) for higher trustworthiness — the app is intentionally conservative rather than optimizing for always producing an answer.
