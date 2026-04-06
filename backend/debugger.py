"""
debugger.py — The Debug Wrapper.

debug_llm_call() is the single entry-point that orchestrates the full
pipeline and captures every artifact for later inspection.
"""

from datetime import datetime

from rag_pipeline import retrieve, build_prompt, call_llm
from evaluator import run_all_evaluations
from analyzer import analyze
from suggestions import generate_suggestions
from database import save_log


def debug_llm_call(
    query: str,
    top_k: int = 3,
    custom_prompt_prefix: str | None = None,
) -> dict:
    """
    Full pipeline with instrumentation.

    Steps:
      1. Retrieve relevant documents (RAG)
      2. Build prompt
      3. Call LLM, measure latency
      4. Evaluate all dimensions
      5. Root-cause analysis
      6. Generate suggestions
      7. Persist structured log
      8. Return full trace

    Args:
        query              — user question
        top_k              — number of docs to retrieve
        custom_prompt_prefix — override system instruction (for testing)

    Returns:
        Full trace dict including all pipeline artifacts and diagnostics.
    """
    # ── Stage 1: Retrieval ────────────────────────────────────────────────
    retrieved_docs = retrieve(query, top_k=top_k)
    scores = [d.get("score", 0.0) for d in retrieved_docs]

    # ── Stage 2: Prompt construction ──────────────────────────────────────
    prompt = build_prompt(query, retrieved_docs)
    if custom_prompt_prefix:
        # Allow callers to inject a different system instruction (test scenarios)
        prompt = custom_prompt_prefix + "\n\n" + prompt

    # ── Stage 3: LLM generation ───────────────────────────────────────────
    response, latency = call_llm(prompt)

    # ── Stage 4: Evaluation ───────────────────────────────────────────────
    evaluations = run_all_evaluations(
        query=query,
        retrieved_docs=retrieved_docs,
        prompt=prompt,
        response=response,
        latency=latency,
    )

    # ── Stage 5: Root-cause analysis ─────────────────────────────────────
    analysis = analyze(evaluations)

    # ── Stage 6: Suggestions ─────────────────────────────────────────────
    suggestions = generate_suggestions(analysis)

    # ── Stage 7: Structured log ──────────────────────────────────────────
    log_entry = {
        # Core artifacts (observable by developer)
        "query": query,
        "retrieved_docs": retrieved_docs,
        "scores": scores,
        "prompt": prompt,
        "response": response,
        "latency": latency,
        "timestamp": datetime.utcnow().isoformat(),
        # Diagnostics
        "evaluations": evaluations,
        "analysis": analysis,
        "suggestions": suggestions,
        # Derived convenience fields for the dashboard
        "status": "pass" if analysis["overall_pass"] else "fail",
        "issue_count": analysis["issue_count"],
        "primary_issue": analysis["primary"]["issue_type"],
    }

    log_id = save_log(log_entry)
    log_entry["id"] = log_id

    return log_entry