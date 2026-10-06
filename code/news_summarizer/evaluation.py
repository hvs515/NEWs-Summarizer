import re

from .classical import clean_text
from .config import CONFIG

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
WORD_RE = re.compile(r"[a-z0-9]+")


def _numbers(text):
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def _words(text):
    return WORD_RE.findall(text.lower())


def evaluate_summary(source, summary, entities):
    source, summary = clean_text(source), clean_text(summary)
    src_words, sum_words = _words(source), _words(summary)

    all_entities = [e for values in entities.values() for e in values]
    covered = [e for e in all_entities if e.lower() in summary.lower()]

    unsupported = sorted(_numbers(summary) - _numbers(source))

    src_set, sum_set = set(src_words), set(sum_words)
    overlap = len(src_set & sum_set)

    return {
        "summary_words": len(sum_words),
        "target_words": CONFIG["SUMMARY_LENGTH_WORDS"],
        "compression_ratio": round(len(sum_words) / max(len(src_words), 1), 3),
        "entity_coverage": round(len(covered) / max(len(all_entities), 1), 3),
        "entities_missing": [e for e in all_entities if e not in covered],
        "unsupported_numbers": unsupported,
        "unigram_precision": round(overlap / max(len(sum_set), 1), 3),
        "unigram_recall": round(overlap / max(len(src_set), 1), 3),
    }
