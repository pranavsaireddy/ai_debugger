# 🔍 AI Debugger

A RAG pipeline observability tool. Feed it documents and queries — it runs the full RAG pipeline, evaluates quality at every stage, detects issues, and surfaces traces + fix suggestions in a dashboard.

> Built to answer: *why did my RAG pipeline give a bad answer?*

---

## Architecture

```
React Frontend (Vite)
        │
        ▼
FastAPI Backend  ←→  OpenAI API
        │             (text-embedding-ada-002 + gpt-3.5-turbo)
        ▼
  data/logs.json
```

---

## Pipeline — 7 Stages Per Query

```
Documents + Query
        │
        ▼
  1. Retrieval          ← chunk → embed → FAISS → top-K
        │
        ▼
  2. Prompt Construction
        │
        ▼
  3. LLM Call           ← gpt-3.5-turbo
        │
        ▼
  4. Evaluation         ← cosine similarity vs. threshold (0.50)
        │
        ▼
  5. Root-Cause Analysis  ← maps to 5 issue types
        │
        ▼
  6. Suggestion Generation
        │
        ▼
  7. Structured Logging   ← atomic write to logs.json
```

---

## What Gets Detected

| Issue Type | Detection Method |
|---|---|
| `RETRIEVAL_FAILURE` | Cosine similarity of retrieved chunks below threshold |
| `HALLUCINATION` | Similarity between LLM answer and source chunks below threshold |
| `PROMPT_QUALITY` | Prompt structure analysis |
| `HIGH_LATENCY` | Latency classification: fast / acceptable / slow |
| `NONE` | Pipeline passed all checks |

---

## Tech Stack

| Layer | Stack |
|---|---|
| **Backend** | FastAPI · Pydantic · OpenAI SDK (`text-embedding-ada-002`, `gpt-3.5-turbo`) · FAISS · NumPy |
| **Storage** | JSON flat-file (`data/logs.json`) — atomic write via `.tmp` → rename |
| **Frontend** | React 19.2.4 · Vite 8.0.1 · Recharts 3.8.1 · ESLint 9.39.4 |

---

## Backend Modules

| Module | Role |
|---|---|
| `main.py` | FastAPI app — 7 endpoints |
| `rag_pipeline.py` | Chunk → embed → FAISS index → top-K retrieval → LLM call |
| `debugger.py` | Orchestrates the full 7-stage pipeline per query |
| `evaluator.py` | Cosine similarity scoring, hallucination detection, latency classification |
| `analyzer.py` | Root-cause analysis — maps evaluation signals to 5 issue types |
| `suggestions.py` | Generates fix suggestions per issue type |
| `database.py` | Atomic read/write to `data/logs.json` |

---

## Frontend — 4 Tabs

Dark terminal-style UI, JetBrains Mono font.

| Tab | Shows |
|---|---|
| **Query List** | All past queries with issue type + latency |
| **Trace View** | Full pipeline trace for a selected query — each stage with inputs/outputs |
| **Issues & Fixes** | Detected issue type + generated fix suggestions |
| **Metrics Dashboard** | LineChart / BarChart / PieChart across all queries |

---

## Setup

### Backend

```bash
cd backend
pip install fastapi uvicorn openai faiss-cpu numpy python-dotenv
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...     # Linux/Mac
set OPENAI_API_KEY=sk-...        # Windows
```

```bash
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
```

---

## Folder Structure

```
ai-debugger/
├── backend/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── debugger.py
│   ├── evaluator.py
│   ├── analyzer.py
│   ├── suggestions.py
│   └── database.py
├── data/
│   └── logs.json          ← auto-created on first query
└── frontend/
    └── src/
        └── App.jsx        ← single-file app, 4 tabs
```

---

## Known Gaps

| Gap | Status |
|---|---|
| `requirements.txt` | Not present — dependencies listed above must be installed manually |
| `.env` handling | No `python-dotenv` setup — API key must be set as environment variable |
| Unit tests | No pytest setup |
| Docker | Not configured |
