"""Align outputs to the complete dataset so missing predictions remain visible."""

from .data import Example
from .metrics import score_summary


def evaluate(examples: list[Example], predictions: list[dict], language: str) -> dict:
    if language not in ("en", "zh"):
        raise ValueError("language must be en or zh")
    expected = {example.id for example in examples}
    by_id = {}
    for row in predictions:
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier not in expected:
            raise ValueError(f"Unknown prediction id: {identifier!r}")
        if identifier in by_id:
            raise ValueError(f"Duplicate prediction id: {identifier!r}")
        status = row.get("status")
        if status not in ("ok", "error"):
            raise ValueError(f"Prediction {identifier!r} requires status 'ok' or 'error'")
        if status == "ok" and (
            not isinstance(row.get("summary"), str) or not row["summary"].strip()
        ):
            raise ValueError(f"Prediction {identifier!r} has an empty or invalid summary")
        by_id[identifier] = row
    per_sample = []
    counts = {"total": len(examples), "successful": 0, "failed": 0,
              "missing": 0, "without_references": 0, "scored": 0}
    for example in examples:
        row = by_id.get(example.id)
        if row is None:
            counts["missing"] += 1
        elif row["status"] == "error":
            counts["failed"] += 1
        else:
            counts["successful"] += 1
            if not example.references:
                counts["without_references"] += 1
                continue
            scores = score_summary(row["summary"], example.references, language)
            per_sample.append({"id": example.id, **scores})
            counts["scored"] += 1
    macro = {
        key: sum(row[key] for row in per_sample) / len(per_sample)
        for key in ("rouge1", "rouge2", "rougeL")
    } if per_sample else None
    return {
        "counts": counts,
        "language": language,
        "score_range": [0, 1],
        "aggregation": "mean F1 across references, then macro mean across scored videos",
        "metric_variant": "custom ROUGE without stemming; en word / zh character-aware tokens",
        "coverage_note": "Scores include successful videos with references only; without_references counts successful unreferenced videos. Inspect counts before comparing runs.",
        "macro_scores": macro,
        "per_sample": per_sample,
    }
