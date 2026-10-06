"""Offline extractive summarizer used as a baseline and as a fallback when no API key is set.

Sentences are scored by the normalised frequency of their content words, with a small
bonus for named entities and for appearing early in the article (lead bias in news).
"""

import re
from collections import Counter

from .classical import clean_text, get_nlp
from .config import CONFIG


def _score_sentences(doc):
    content = [
        t.lemma_.lower()
        for t in doc
        if t.is_alpha and not t.is_stop and len(t) > 2
    ]
    freq = Counter(content)
    if not freq:
        return []
    top = max(freq.values())

    sents = list(doc.sents)
    scored = []
    for idx, sent in enumerate(sents):
        tokens = [t.lemma_.lower() for t in sent if t.is_alpha and not t.is_stop and len(t) > 2]
        if not tokens:
            continue
        score = sum(freq[t] / top for t in tokens) / len(tokens) ** 0.5
        score += 0.1 * len(sent.ents)
        score += 0.3 * (1 - idx / len(sents))
        scored.append((score, idx, sent.text.strip()))
    return scored


def extractive_summary(text, max_words=None, num_points=None):
    """Return a dict with the same keys as the LLM output: summary, key_points, headline."""
    max_words = max_words or CONFIG["SUMMARY_LENGTH_WORDS"]
    num_points = num_points or CONFIG["NUM_KEY_POINTS"]
    doc = get_nlp()(clean_text(text))
    scored = _score_sentences(doc)
    if not scored:
        return {"summary": "", "key_points": [], "headline": ""}

    ranked = sorted(scored, key=lambda s: s[0], reverse=True)

    chosen, words = [], 0
    for score, idx, sent in ranked:
        n = len(sent.split())
        if chosen and words + n > max_words:
            continue
        chosen.append((idx, sent))
        words += n
    summary = " ".join(sent for _, sent in sorted(chosen))

    key_points = [sent for _, _, sent in sorted(ranked[:num_points], key=lambda s: s[1])]

    lead = scored[0][2]
    return {"summary": summary, "key_points": key_points, "headline": make_headline(lead)}


def make_headline(sentence, max_words=18):
    """Turn the lead sentence into a headline: drop the first appositive, keep the main clause."""
    text = re.sub(r",[^,]+,", "", sentence, count=1)
    text = text.split(",")[0].strip().rstrip(".")
    words = text.split()
    return " ".join(words[:max_words])
