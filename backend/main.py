"""
main.py — FastAPI backend for the AI System Debugger.

Endpoints:
  POST /ingest          — Load documents into the vector store
  POST /query           — Run a query through the full debug pipeline
  GET  /logs            — Retrieve all debug logs (newest first)
  GET  /logs/{id}       — Retrieve a single log trace
  GET  /metrics         — Aggregate dashboard metrics
  DELETE /logs          — Clear all logs
  GET  /health          — Liveness probe
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from debugger import debug_llm_call
from rag_pipeline import ingest_documents
from database import get_all_logs, get_log, clear_logs

app = FastAPI(
    title="AI System Debugger",
    description="Production-grade observability for RAG/LLM pipelines.",
    version="1.0.0",
)

# Allow the React dev server (port 3000) and production builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    documents: list[dict] = Field(
        ...,
        example=[{"title": "RAG Overview", "content": "RAG combines retrieval..."}],
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=20)
    custom_prompt_prefix: str | None = None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", summary="Ingest documents into FAISS vector store")
def ingest(req: IngestRequest):
    stats = ingest_documents(req.documents)
    return {"success": True, "stats": stats}


@app.post("/query", summary="Debug a query through the full RAG pipeline")
def query(req: QueryRequest):
    result = debug_llm_call(
        query=req.query,
        top_k=req.top_k,
        custom_prompt_prefix=req.custom_prompt_prefix,
    )
    return result


@app.get("/logs", summary="List all debug logs")
def list_logs():
    return {"logs": get_all_logs()}


@app.get("/logs/{log_id}", summary="Get a single debug trace")
def get_single_log(log_id: str):
    log = get_log(log_id)
    if not log:
        raise HTTPException(404, detail="Log not found")
    return log


@app.delete("/logs", summary="Clear all logs")
def delete_logs():
    count = clear_logs()
    return {"deleted": count}


@app.get("/metrics", summary="Aggregate metrics for the dashboard")
def metrics():
    """
    Compute summary stats from all stored logs.
    Used by the dashboard Metrics tab.
    """
    logs = get_all_logs()
    if not logs:
        return {"total": 0}

    total = len(logs)
    failures = [l for l in logs if l.get("status") == "fail"]
    hallucinations = [
        l for l in logs
        if l.get("evaluations", {}).get("hallucination", {}).get("hallucination_risk")
    ]
    latencies = [l.get("latency", 0.0) for l in logs if l.get("latency") is not None]
    retrieval_scores = [
        l.get("evaluations", {}).get("retrieval", {}).get("avg_score", 0.0)
        for l in logs
    ]

    # Issue type breakdown
    issue_counts: dict[str, int] = {}
    for log in logs:
        pt = log.get("primary_issue", "NONE")
        issue_counts[pt] = issue_counts.get(pt, 0) + 1

    # Latency over time (for chart)
    latency_series = [
        {"timestamp": l.get("timestamp", ""), "latency": l.get("latency", 0)}
        for l in reversed(logs)  # chronological order
    ]

    return {
        "total": total,
        "failures": len(failures),
        "error_rate": round(len(failures) / total, 3),
        "hallucination_count": len(hallucinations),
        "hallucination_rate": round(len(hallucinations) / total, 3),
        "avg_latency": round(float(np.mean(latencies)), 3) if latencies else 0,
        "avg_retrieval_score": round(float(np.mean(retrieval_scores)), 3) if retrieval_scores else 0,
        "issue_breakdown": issue_counts,
        "latency_series": latency_series[-50:],  # last 50 queries
    }