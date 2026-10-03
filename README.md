# Movie Intelligence & Follow-up Assistant

An AI-powered application that answers questions about movie subtitles, retrieves relevant dialogue with timestamp citations, and sends movie information via email.  
Built for the **AI KYRO** internship take-home assignment.

---

## Overview

The application indexes movie subtitle files (`.srt`) into a vector database and lets you ask natural-language questions about them.  It routes each request through an intelligent agent that decides whether to:

- **Answer informational questions** using RAG (Retrieval-Augmented Generation)
- **Send an email** with AI-generated content and timestamp citations
- **Ask for clarification** when the request is ambiguous or incomplete

All LLM inference runs locally via **Ollama** — no paid APIs required.

---

## Architecture

```
User Request
    │
    ▼
AgentRouter          ← detects intent (informational / email / clarification)
    │
    ▼
AmbiguityHandler     ← detects multi-movie ambiguity in retrieved results
    │
    ├─── informational ──► RAG Retriever → LocalLLM → Answer + Citations
    │
    ├─── email ──────────► RAG Retriever → LocalLLM → MCP Email Tool
    │
    └─── clarification ──► Returns clarification question (no RAG / MCP)
```

### Component Map

| Path | Purpose |
|---|---|
| `app/agent/agent.py` | Top-level `MovieAgent` — orchestrates the full pipeline |
| `app/agent/router.py` | `AgentRouter` — classifies intent, extracts recipient/query |
| `app/agent/ambiguity.py` | `AmbiguityHandler` — detects multi-movie retrieval |
| `app/agent/llm.py` | `LocalLLM` — calls local Ollama HTTP API with retry |
| `app/rag/parser.py` | SRT subtitle parser |
| `app/rag/chunker.py` | Sliding-window subtitle chunker with metadata |
| `app/rag/embeddings.py` | `EmbeddingModel` — sentence-transformers wrapper |
| `app/rag/vector_store.py` | `VectorStore` — ChromaDB persistence |
| `app/rag/retriever.py` | `Retriever` — semantic search |
| `app/rag/answerer.py` | `RAGAnswerer` — builds prompt + calls LLM |
| `app/mcp/email_server.py` | MCP server with `send_email` tool |
| `app/mcp/email_sender.py` | `EmailSender` — mock transport (development) |
| `app.py` | Streamlit UI entry point |

---

## Requirements

- **Python** 3.11+
- **Ollama** (running locally)  
  Install from https://ollama.com
- **Model**: `qwen3.5:9b`

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/movie-intelligence-agent.git
cd movie-intelligence-agent

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the Ollama model
ollama pull qwen3.5:9b
```

---

## Indexing Subtitles

Place `.srt` files in the `subtitles/` directory, then run the ingestion script (if present) or use the `VectorStore` + `RAGAnswerer` pipeline directly.

The project ships with a pre-built `chroma_db/` for *A Bucket Of Blood (1959)*.

---

## Running the Application

### Streamlit UI

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

The UI lets you:
- Select a movie from the sidebar (or search all)
- Ask informational questions
- Send email requests (mock transport)
- See intent, answer, and timestamp citations

### Running tests

```bash
# End-to-end agent tests (Phase 10)
python -m scripts.test_agent

# Individual component tests
python -m scripts.test_llm
python -m scripts.test_rag
python -m scripts.test_router
python -m scripts.test_ambiguity
python -m scripts.test_mcp_email
python -m scripts.test_retriever
python -m scripts.test_embeddings
python -m scripts.test_chunker
python -m scripts.test_vector_store
python -m scripts.test_parser
```

---

## RAG Workflow

```
1. User asks:  "What does the artist say about art?"
2. Retriever   embeds the query → searches ChromaDB → returns top-N chunks
3. RAGAnswerer builds a prompt with SOURCE 1…N blocks
4. LocalLLM    generates an answer with in-text [1][3] citations
5. Agent       attaches exact movie + timestamp metadata to citations
6. UI / test   displays answer + citation blocks
```

Citations are constructed from retrieved metadata — the LLM is explicitly instructed **not** to invent timestamps.

---

## Agent Routing Workflow

The `AgentRouter` uses regex patterns to classify each request:

| Signal | Intent |
|---|---|
| No email action verb | `informational` |
| Email verb + email address + substantive topic | `email` |
| Email verb + vague topic only ("the scene", "the dialogue") | `clarification` |
| Email verb + missing email address | `clarification` |
| Empty request | `clarification` |

---

## Ambiguity Handling

After retrieval, `AmbiguityHandler` checks whether results span multiple movies.  If more than one movie is found and the user did not select a specific movie, a clarification question is returned:

> "I found relevant information in multiple movies. Which movie are you referring to? Possible matches: Movie A, Movie B."

---

## MCP Email Workflow

The email tool is implemented as an MCP (Model Context Protocol) server.

```
MovieAgent.send_email()
    └─► MCP Client
           └─► email_server.send_email()
                  └─► EmailSender.send()  [mock transport]
                         └─► returns { success, status, recipient, subject, body }
```

**Development mode**: `EMAIL_DRY_RUN=true` in `.env` — no real emails are sent.  
The mock transport validates all fields and returns a success response.

To enable real SMTP, configure `.env` (see `.env.example`) and update `EmailSender`.

---

## Example Questions

```
What does the artist say about art?
Why does Walter become important to the artists?
What happens when Walter shows his sculpture?
Who is Naolia and what does she want?
Email me a breakdown of what the artist says about art to you@example.com
```

---

## Known Limitations

- Only `.srt` subtitle files are supported.
- The `qwen3.5:9b` thinking model occasionally returns empty content on the first attempt; the `LocalLLM` retries up to 3 times automatically.
- Email uses a **mock transport** only — no real emails are delivered.
- Ambiguity detection is based on distinct movie titles in retrieval results, not semantic similarity.
- The Streamlit UI is intentionally minimal (assignment spec: "no polish needed").

---

## Project Structure

```
movie-intelligence-agent/
├── app/
│   ├── agent/
│   │   ├── agent.py        # MovieAgent (orchestrator)
│   │   ├── router.py       # AgentRouter
│   │   ├── ambiguity.py    # AmbiguityHandler
│   │   └── llm.py          # LocalLLM (Ollama)
│   ├── rag/
│   │   ├── parser.py       # SRT parser
│   │   ├── chunker.py      # Subtitle chunker
│   │   ├── embeddings.py   # Sentence-transformer embeddings
│   │   ├── vector_store.py # ChromaDB wrapper
│   │   ├── retriever.py    # Semantic retriever
│   │   └── answerer.py     # RAG answerer
│   └── mcp/
│       ├── email_server.py # MCP email server
│       └── email_sender.py # Email transport (mock)
├── scripts/
│   ├── test_agent.py       # Phase 10 end-to-end tests
│   └── test_*.py           # Component tests
├── subtitles/              # .srt files
├── chroma_db/              # ChromaDB persistent store
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
└── README.md
```
