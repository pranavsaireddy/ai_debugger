"""
evaluator.py — Measurable quality metrics for each pipeline stage.

All metrics return numeric scores so they can be tracked over time
and compared across queries. No vague heuristics.
"""

import re
import numpy as np
from typing import Optional
from rag_pipeline import embed_texts

# ---------------------------------------------------------------------------
# Thresholds (tune per deployment)
# ---------------------------------------------------------------------------

RETRIEVAL_SIMILARITY_THRESHOLD = 0.50   # below → weak retrieval
HALLUCINATION_SIMILARITY_THRESHOLD = 0.60  # below → possible hallucination
MAX_PROMPT_TOKENS_ESTIMATE = 3000       # rough char/4 estimate


# ---------------------------------------------------------------------------
# 1. Retrieval Quality
# ---------------------------------------------------------------------------

def evaluate_retrieval(retrieved_docs: list[dict]) -> dict:
    """
    Assess retrieval quality from similarity scores.

    Returns:
        avg_score       — mean cosine similarity across retrieved docs
        min_score       — worst-case doc similarity
        doc_count       — number of docs retrieved
        retrieval_ok    — True if avg_score ≥ threshold
        flag            — human-readable status string
    """
    if not retrieved_docs:
        return {
            "avg_score": 0.0, "min_score": 0.0,
            "doc_count": 0, "retrieval_ok": False,
            "flag": "NO_DOCS_RETRIEVED",
        }

    scores = [d.get("score", 0.0) for d in retrieved_docs]
    avg = float(np.mean(scores))
    minimum = float(np.min(scores))
    ok = avg >= RETRIEVAL_SIMILARITY_THRESHOLD

    return {
        "avg_score": round(avg, 4),
        "min_score": round(minimum, 4),
        "doc_count": len(retrieved_docs),
        "retrieval_ok": ok,
        "flag": "OK" if ok else "LOW_SIMILARITY",
    }


# ---------------------------------------------------------------------------
# 2. Prompt Quality
# ---------------------------------------------------------------------------

def evaluate_prompt(prompt: str) -> dict:
    """
    Static checks on prompt structure.

    Checks:
        has_context          — contains the word 'context'
        has_instruction      — contains an instruction keyword
        token_estimate       — rough token count (chars / 4)
        within_token_limit   — under MAX_PROMPT_TOKENS_ESTIMATE
        score                — 0-100 composite
        flag
    """
    has_context = "context" in prompt.lower()
    instruction_keywords = ["answer", "explain", "describe", "summarize", "do not", "only"]
    has_instruction = any(kw in prompt.lower() for kw in instruction_keywords)
    has_grounding = any(kw in prompt.lower() for kw in ["only", "do not fabricate", "based on"])

    token_estimate = len(prompt) // 4
    within_limit = token_estimate <= MAX_PROMPT_TOKENS_ESTIMATE

    # Weighted score
    score = (
        int(has_context) * 35
        + int(has_instruction) * 35
        + int(has_grounding) * 20
        + int(within_limit) * 10
    )

    flags = []
    if not has_context:
        flags.append("MISSING_CONTEXT_REFERENCE")
    if not has_instruction:
        flags.append("MISSING_INSTRUCTION")
    if not has_grounding:
        flags.append("MISSING_GROUNDING_CONSTRAINT")
    if not within_limit:
        flags.append("PROMPT_TOO_LONG")

    return {
        "has_context": has_context,
        "has_instruction": has_instruction,
        "has_grounding": has_grounding,
        "token_estimate": token_estimate,
        "within_token_limit": within_limit,
        "score": score,
        "flag": ", ".join(flags) if flags else "OK",
    }


# ---------------------------------------------------------------------------
# 3. Hallucination Detection
# ---------------------------------------------------------------------------

def evaluate_hallucination(
    response: str,
    retrieved_docs: list[dict],
    context_sentences: Optional[list[str]] = None,
) -> dict:
    """
    Semantic overlap between LLM output and retrieved context.

    Strategy:
      1. Embed the full response.
      2. Embed the concatenated context.
      3. Cosine similarity (already L2-normalised → dot-product).
      4. Sentence-level grounding: what fraction of response sentences
         have a nearest-neighbour context score ≥ threshold?

    Returns:
        response_context_similarity   — overall semantic overlap [0,1]
        grounded_sentence_ratio       — fraction of sentences with good match
        hallucination_risk            — True = likely hallucination
        flag
    """
    if not retrieved_docs or not response.strip():
        return {
            "response_context_similarity": 0.0,
            "grounded_sentence_ratio": 0.0,
            "hallucination_risk": True,
            "flag": "CANNOT_EVALUATE",
        }

    context_text = " ".join(d.get("text", "") for d in retrieved_docs)

    # Overall similarity
    vecs = embed_texts([response, context_text])
    overall_sim = float(np.dot(vecs[0], vecs[1]))

    # Sentence-level grounding
    resp_sentences = [s.strip() for s in re.split(r'[.!?]', response) if len(s.strip()) > 15]
    ctx_sentences  = [s.strip() for s in re.split(r'[.!?]', context_text) if len(s.strip()) > 15]

    grounded_ratio = 0.0
    if resp_sentences and ctx_sentences:
        resp_vecs = embed_texts(resp_sentences)
        ctx_vecs  = embed_texts(ctx_sentences)
        sim_matrix = resp_vecs @ ctx_vecs.T  # (R, C)
        max_sims = sim_matrix.max(axis=1)    # best context match per sentence
        grounded_ratio = float((max_sims >= HALLUCINATION_SIMILARITY_THRESHOLD).mean())

    hallucination_risk = (
        overall_sim < HALLUCINATION_SIMILARITY_THRESHOLD
        or grounded_ratio < 0.5
    )

    return {
        "response_context_similarity": round(overall_sim, 4),
        "grounded_sentence_ratio": round(grounded_ratio, 4),
        "hallucination_risk": hallucination_risk,
        "flag": "POSSIBLE_HALLUCINATION" if hallucination_risk else "OK",
    }


# ---------------------------------------------------------------------------
# 4. Latency
# ---------------------------------------------------------------------------

def evaluate_latency(latency: float) -> dict:
    """
    Classify response latency into severity buckets.

    Returns:
        latency_seconds
        tier       — FAST / ACCEPTABLE / SLOW / CRITICAL
        flag
    """
    if latency < 1.0:
        tier, flag = "FAST", "OK"
    elif latency < 3.0:
        tier, flag = "ACCEPTABLE", "OK"
    elif latency < 7.0:
        tier, flag = "SLOW", "HIGH_LATENCY"
    else:
        tier, flag = "CRITICAL", "CRITICAL_LATENCY"

    return {
        "latency_seconds": round(latency, 3),
        "tier": tier,
        "flag": flag,
    }


# ---------------------------------------------------------------------------
# Composite runner
# ---------------------------------------------------------------------------

def run_all_evaluations(
    query: str,
    retrieved_docs: list[dict],
    prompt: str,
    response: str,
    latency: float,
) -> dict:
    """Run every evaluator and return a single merged dict."""
    return {
        "retrieval": evaluate_retrieval(retrieved_docs),
        "prompt": evaluate_prompt(prompt),
        "hallucination": evaluate_hallucination(response, retrieved_docs),
        "latency": evaluate_latency(latency),
    }