"""Small, deterministic ROUGE F1 metrics for reference-based summaries.

Scores are in [0, 1]. Multiple references are scored independently, then their
scores are averaged for each video; callers can macro-average videos. This is
not the best-reference aggregation used by some benchmark implementations.

English tokens are lowercase ASCII letters/digits, with internal apostrophes
retained. Chinese uses individual CJK characters plus English word/number
tokens. Punctuation is ignored and no stemming is performed. These explicit
tokenizers make this baseline reproducible, but results need not match other
ROUGE packages or benchmark tokenization. ROUGE measures reference overlap,
not factual correctness, temporal coverage, or hallucinations.
"""

from __future__ import annotations

from collections import Counter
import re


_EN_WORD = r"[a-z0-9]+(?:'[a-z0-9]+)*"
_TOKENIZERS = {
    "en": re.compile(_EN_WORD),
    "zh": re.compile(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
        r"\U00020000-\U0002fa1f]|" + _EN_WORD
    ),
}


def _ngram_f1(prediction: list[str], reference: list[str], n: int) -> float:
    pred_counts = Counter(
        tuple(prediction[i : i + n]) for i in range(len(prediction) - n + 1)
    )
    ref_counts = Counter(
        tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)
    )
    denominator = sum(pred_counts.values()) + sum(ref_counts.values())
    if not denominator:
        return 0.0
    overlap = sum((pred_counts & ref_counts).values())
    return 2.0 * overlap / denominator


def _lcs_length(first: list[str], second: list[str]) -> int:
    """Compute LCS in O(len(first) * len(second)) time and O(min length) space."""
    if len(first) < len(second):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for first_token in first:
        current = [0]
        for index, second_token in enumerate(second, start=1):
            if first_token == second_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def score_summary(
    prediction: str, references: list[str], language: str = "en"
) -> dict[str, float]:
    """Return mean-reference ROUGE-1/2/L F1 for a single prediction.

    ``language`` selects tokenization, and must be ``en`` or ``zh``. Empty
    predictions score zero. Empty or whitespace-only references are rejected;
    references consisting only of ignored characters score zero. A reference
    and prediction shorter than two tokens have ROUGE-2 zero, even if equal.
    """
    if language not in _TOKENIZERS:
        raise ValueError("language must be 'en' or 'zh'")
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if isinstance(references, str) or not references:
        raise ValueError("references must be a nonempty list of strings")
    if any(not isinstance(reference, str) or not reference.strip() for reference in references):
        raise ValueError("references must contain only nonblank strings")

    tokenizer = _TOKENIZERS[language]
    prediction_tokens = tokenizer.findall(prediction.lower())
    scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for reference in references:
        reference_tokens = tokenizer.findall(reference.lower())
        scores["rouge1"] += _ngram_f1(prediction_tokens, reference_tokens, 1)
        scores["rouge2"] += _ngram_f1(prediction_tokens, reference_tokens, 2)
        denominator = len(prediction_tokens) + len(reference_tokens)
        if denominator:
            scores["rougeL"] += (
                2.0 * _lcs_length(prediction_tokens, reference_tokens) / denominator
            )
    return {name: value / len(references) for name, value in scores.items()}
