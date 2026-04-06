"""
analyzer.py — Root Cause Analyzer.

Converts raw evaluation metrics into structured failure diagnoses.
Each rule maps a metric condition to an issue type + confidence score.
"""

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Issue taxonomy
# ---------------------------------------------------------------------------

ISSUE_RETRIEVAL      = "RETRIEVAL_FAILURE"
ISSUE_PROMPT         = "PROMPT_QUALITY"
ISSUE_HALLUCINATION  = "HALLUCINATION"
ISSUE_LATENCY        = "HIGH_LATENCY"
ISSUE_NONE           = "NONE"


@dataclass
class Issue:
    issue_type: str
    confidence: float          # 0.0–1.0
    description: str
    affected_stage: str        # retrieval | prompt | generation | infra

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rule-based analyzer
# ---------------------------------------------------------------------------

def analyze(evaluations: dict) -> dict:
    """
    Apply diagnostic rules to evaluation outputs.

    Rules (in priority order):
    1. Retrieval avg_score < 0.50   → RETRIEVAL_FAILURE   (high confidence)
    2. Prompt score < 70            → PROMPT_QUALITY       (medium confidence)
    3. hallucination_risk == True   → HALLUCINATION        (scaled by sim score)
    4. Latency tier SLOW/CRITICAL   → HIGH_LATENCY         (informational)

    Returns:
        issues       — list of Issue dicts (highest confidence first)
        primary      — top Issue
        overall_pass — bool
    """
    issues: list[Issue] = []
    ret  = evaluations.get("retrieval", {})
    prm  = evaluations.get("prompt", {})
    hal  = evaluations.get("hallucination", {})
    lat  = evaluations.get("latency", {})

    # Rule 1: Retrieval quality
    avg_score = ret.get("avg_score", 1.0)
    if not ret.get("retrieval_ok", True):
        confidence = round(1.0 - (avg_score / 0.50), 2)  # worse score → higher confidence
        confidence = max(0.1, min(confidence, 1.0))
        issues.append(Issue(
            issue_type=ISSUE_RETRIEVAL,
            confidence=confidence,
            description=(
                f"Retrieved documents have low average similarity ({avg_score:.3f}). "
                "The vector store may not contain relevant content, or the query "
                "is outside the indexed domain."
            ),
            affected_stage="retrieval",
        ))

    # Rule 2: Prompt quality
    prompt_score = prm.get("score", 100)
    if prompt_score < 70:
        confidence = round((70 - prompt_score) / 70, 2)
        issues.append(Issue(
            issue_type=ISSUE_PROMPT,
            confidence=confidence,
            description=(
                f"Prompt quality score is {prompt_score}/100. "
                f"Missing elements: {prm.get('flag', 'unknown')}."
            ),
            affected_stage="prompt",
        ))

    # Rule 3: Hallucination
    if hal.get("hallucination_risk", False):
        sim = hal.get("response_context_similarity", 0.0)
        grounded = hal.get("grounded_sentence_ratio", 0.0)
        confidence = round(1.0 - ((sim + grounded) / 2), 2)
        confidence = max(0.1, min(confidence, 1.0))
        issues.append(Issue(
            issue_type=ISSUE_HALLUCINATION,
            confidence=confidence,
            description=(
                f"Response has low semantic overlap with retrieved context "
                f"(similarity={sim:.3f}, grounded_ratio={grounded:.2%}). "
                "The model may be relying on parametric knowledge rather than context."
            ),
            affected_stage="generation",
        ))

    # Rule 4: Latency
    if lat.get("tier") in ("SLOW", "CRITICAL"):
        issues.append(Issue(
            issue_type=ISSUE_LATENCY,
            confidence=0.9,
            description=(
                f"Response latency is {lat.get('latency_seconds', '?')}s "
                f"(tier={lat.get('tier')}). "
                "Consider caching embeddings or switching to a faster model."
            ),
            affected_stage="infra",
        ))

    # Sort by confidence descending
    issues.sort(key=lambda x: x.confidence, reverse=True)

    primary = issues[0].to_dict() if issues else {
        "issue_type": ISSUE_NONE,
        "confidence": 1.0,
        "description": "No significant issues detected.",
        "affected_stage": "none",
    }

    return {
        "issues": [i.to_dict() for i in issues],
        "primary": primary,
        "overall_pass": len(issues) == 0,
        "issue_count": len(issues),
    }