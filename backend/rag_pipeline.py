"""
rag_pipeline.py — Full Retrieval-Augmented Generation pipeline.

Architecture:
  Documents → Chunker → Embedder → FAISS Index
  Query     → Embedder → FAISS Search → Top-K Docs → Prompt Builder
"""

import re
import time
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy deps — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


VECTOR_STORE_PATH = Path(__file__).parent.parent / "vector_store"
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # character overlap between chunks
TOP_K = 3                 # number of docs to retrieve by default
EMBED_DIM = 1536          # text-embedding-ada-002 dimensionality


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_openai_client() -> Optional["OpenAI"]:
    import os
    key = os.getenv("OPENAI_API_KEY")
    if not key or not OPENAI_AVAILABLE:
        return None
    return OpenAI(api_key=key)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Return (N, EMBED_DIM) float32 matrix.
    Falls back to deterministic random embeddings when OpenAI is unavailable,
    so the rest of the pipeline still runs and scores are plausible.
    """
    client = _get_openai_client()
    if client:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    else:
        # Reproducible fake embeddings — same text → same vector
        rng_seeds = [int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**31) for t in texts]
        vecs = np.array(
            [np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)
             for seed in rng_seeds]
        )
    # L2-normalise so dot-product == cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size character windows."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += size - overlap
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Vector store management
# ---------------------------------------------------------------------------

class FAISSStore:
    """Simple in-memory FAISS index with optional disk persistence."""

    def __init__(self):
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.index: Optional["faiss.IndexFlatIP"] = None

    def build(self, chunks: list[str], meta: list[dict]) -> None:
        """Embed chunks and build the index."""
        if not chunks:
            return
        vecs = embed_texts(chunks)
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(EMBED_DIM)
            self.index.add(vecs)
        else:
            self.index = vecs  # fall back: raw matrix dot-product
        self.chunks = chunks
        self.metadata = meta

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Return top-K results with chunk text, score, and metadata."""
        if self.index is None or not self.chunks:
            return []

        q_vec = embed_texts([query])

        if FAISS_AVAILABLE:
            scores, idxs = self.index.search(q_vec, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                results.append({
                    "text": self.chunks[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx],
                })
        else:
            # Manual cosine similarity via dot-product (vecs already normalised)
            sims = (self.index @ q_vec.T).flatten()
            top_idxs = np.argsort(sims)[::-1][:top_k]
            results = [
                {"text": self.chunks[i], "score": float(sims[i]), "metadata": self.metadata[i]}
                for i in top_idxs
            ]
        return results


# Global singleton store (reset per process)
_store = FAISSStore()


def ingest_documents(documents: list[dict]) -> dict:
    """
    Ingest a list of {title, content} dicts.
    Returns ingestion stats.
    """
    all_chunks, all_meta = [], []
    for doc in documents:
        chunks = chunk_text(doc.get("content", ""))
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"source": doc.get("title", "unknown"), "chunk_index": i})

    _store.build(all_chunks, all_meta)
    return {"documents": len(documents), "chunks": len(all_chunks)}


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-K chunks relevant to query."""
    return _store.search(query, top_k)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain enough information, say "
    "'I don't know based on the provided context.' Do not fabricate information."
)


def build_prompt(query: str, retrieved_docs: list[dict]) -> str:
    """Assemble the final prompt string from query + retrieved chunks."""
    context_blocks = "\n\n".join(
        f"[Source: {d['metadata'].get('source', '?')}]\n{d['text']}"
        for d in retrieved_docs
    )
    return (
        f"SYSTEM: {SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_blocks}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> tuple[str, float]:
    """
    Call the LLM and return (response_text, latency_seconds).
    Degrades to an echo/mock response if OpenAI is unavailable.
    """
    start = time.perf_counter()
    client = _get_openai_client()

    if client:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        text = resp.choices[0].message.content
    else:
        # Mock response for demo / testing without API key
        import hashlib, time as _t
        _t.sleep(0.3)  # simulate latency
        seed = int(hashlib.md5(prompt[:80].encode()).hexdigest(), 16) % 100
        text = (
            "Based on the provided context, " +
            ["the answer relates to the topic described in the documents.",
             "further information is not available in the current context.",
             "the system identifies the relevant information in the retrieved passages.",
             "I cannot find a definitive answer in the provided context."][seed % 4]
        )

    latency = time.perf_counter() - start
    return text, latency