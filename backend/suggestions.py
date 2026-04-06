"""
suggestions.py — Actionable fix suggestions based on diagnosed issues.

Each suggestion is specific, developer-actionable, and mapped to the
exact issue type returned by the analyzer.
"""

from analyzer import (
    ISSUE_RETRIEVAL, ISSUE_PROMPT, ISSUE_HALLUCINATION,
    ISSUE_LATENCY, ISSUE_NONE,
)

# ---------------------------------------------------------------------------
# Suggestion catalog
# ---------------------------------------------------------------------------

_SUGGESTIONS: dict[str, list[dict]] = {
    ISSUE_RETRIEVAL: [
        {
            "action": "Increase top_k retrieval count",
            "detail": "Raise TOP_K from 3 → 5-10 to give the model more candidate passages.",
            "code_hint": "retrieve(query, top_k=8)",
            "priority": "HIGH",
        },
        {
            "action": "Re-chunk documents with smaller chunks",
            "detail": (
                "Reduce CHUNK_SIZE (e.g. 500 → 200 chars) so embeddings are more focused "
                "and similarity scores are higher for narrow questions."
            ),
            "code_hint": "chunk_text(doc, size=200, overlap=30)",
            "priority": "HIGH",
        },
        {
            "action": "Upgrade embedding model",
            "detail": (
                "Switch from text-embedding-ada-002 to text-embedding-3-large "
                "for ~10-20% better retrieval accuracy on technical content."
            ),
            "code_hint": "model='text-embedding-3-large'",
            "priority": "MEDIUM",
        },
        {
            "action": "Add query expansion / HyDE",
            "detail": (
                "Generate a hypothetical answer to the query and embed that instead "
                "(Hypothetical Document Embeddings). Narrows the gap between query "
                "and document distribution."
            ),
            "code_hint": "# Generate: 'A document that would answer: {query}' and embed it",
            "priority": "MEDIUM",
        },
        {
            "action": "Ingest more / better documents",
            "detail": (
                "If similarity scores are universally low, the knowledge base may not "
                "cover this topic. Add relevant documents and re-index."
            ),
            "code_hint": "ingest_documents([{'title': '...', 'content': '...'}])",
            "priority": "HIGH",
        },
    ],
    ISSUE_PROMPT: [
        {
            "action": "Add explicit context reference",
            "detail": "Include 'Use ONLY the provided context' in the system message.",
            "code_hint": "SYSTEM_PROMPT = 'Answer using ONLY the context below...'",
            "priority": "HIGH",
        },
        {
            "action": "Add grounding instruction",
            "detail": (
                "Append: 'If the context does not contain the answer, say "
                "\"I don't know\". Do not fabricate.' to prevent hallucination."
            ),
            "code_hint": "prompt += '\\nDo NOT invent information not present in the context.'",
            "priority": "HIGH",
        },
        {
            "action": "Add output format constraint",
            "detail": (
                "Specify answer structure: 'Answer in 3 sentences or fewer.' "
                "Reduces rambling and off-topic content."
            ),
            "code_hint": "SYSTEM_PROMPT += ' Answer concisely in ≤3 sentences.'",
            "priority": "MEDIUM",
        },
        {
            "action": "Trim context to fit token budget",
            "detail": (
                "If prompt is over 3 000 tokens, truncate retrieved chunks "
                "or switch to a 16k/128k context model."
            ),
            "code_hint": "context = context[:4000]  # chars ≈ 1 000 tokens",
            "priority": "MEDIUM",
        },
    ],
    ISSUE_HALLUCINATION: [
        {
            "action": "Add grounding constraint to prompt",
            "detail": (
                "Explicitly forbid the model from using outside knowledge: "
                "'Answer ONLY from the context. Do not use prior knowledge.'"
            ),
            "code_hint": "SYSTEM_PROMPT = 'Use only the context. Do not use prior knowledge.'",
            "priority": "HIGH",
        },
        {
            "action": "Lower LLM temperature",
            "detail": (
                "Set temperature=0.0 or 0.1 to reduce creative generation "
                "and anchor responses closer to retrieved text."
            ),
            "code_hint": "temperature=0.0",
            "priority": "HIGH",
        },
        {
            "action": "Use post-generation grounding check",
            "detail": (
                "After generation, embed each response sentence and verify its "
                "cosine similarity to context. Reject and retry if below 0.6."
            ),
            "code_hint": "# Sentence-level similarity check before returning to user",
            "priority": "MEDIUM",
        },
        {
            "action": "Improve retrieval quality first",
            "detail": (
                "Hallucinations often stem from poor retrieval. Fix retrieval "
                "issues first — if the context is strong, the model has less "
                "incentive to improvise."
            ),
            "code_hint": "# See RETRIEVAL_FAILURE suggestions",
            "priority": "HIGH",
        },
    ],
    ISSUE_LATENCY: [
        {
            "action": "Cache embeddings",
            "detail": "Pre-compute and store embeddings for frequently asked queries.",
            "code_hint": "# Use Redis or an on-disk cache keyed by query hash",
            "priority": "HIGH",
        },
        {
            "action": "Switch to a faster model",
            "detail": "Use gpt-3.5-turbo instead of gpt-4 for non-critical queries.",
            "code_hint": "model='gpt-3.5-turbo'",
            "priority": "HIGH",
        },
        {
            "action": "Enable streaming",
            "detail": (
                "Use OpenAI streaming API to start showing tokens immediately "
                "— perceived latency drops significantly."
            ),
            "code_hint": "stream=True",
            "priority": "MEDIUM",
        },
    ],
    ISSUE_NONE: [
        {
            "action": "No action required",
            "detail": "Pipeline is operating within all quality thresholds.",
            "code_hint": "",
            "priority": "INFO",
        }
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_suggestions(analysis: dict) -> list[dict]:
    """
    Given the output of analyzer.analyze(), return an ordered list of
    actionable suggestions (highest priority issues first).
    """
    suggestions = []
    seen_actions: set[str] = set()

    for issue in analysis.get("issues", []):
        issue_type = issue.get("issue_type", ISSUE_NONE)
        for sug in _SUGGESTIONS.get(issue_type, []):
            if sug["action"] not in seen_actions:
                suggestions.append({
                    **sug,
                    "related_issue": issue_type,
                    "issue_confidence": issue.get("confidence"),
                })
                seen_actions.add(sug["action"])

    if not suggestions:
        suggestions = _SUGGESTIONS[ISSUE_NONE]

    return suggestions