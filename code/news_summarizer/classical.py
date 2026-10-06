import re
from functools import lru_cache

from .config import CONFIG

ENTITY_LABELS = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Locations",
    "LOC": "Locations",
    "DATE": "Dates",
    "TIME": "Dates",
}


@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    for name in CONFIG["SPACY_MODELS"]:
        try:
            return spacy.load(name)
        except OSError:
            continue
    raise OSError(
        "No spaCy English model found. Run: python -m spacy download en_core_web_sm"
    )


@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def sentiment_label(compound):
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def extract_entities(doc):
    entities = {"People": [], "Organizations": [], "Locations": [], "Dates": []}
    for ent in doc.ents:
        category = ENTITY_LABELS.get(ent.label_)
        if category:
            entities[category].append(ent.text)
    return {category: sorted(set(values)) for category, values in entities.items()}


def analyze_classical_nlp(text):
    cleaned = clean_text(text)
    doc = get_nlp()(cleaned)

    sentences = [sent.text for sent in doc.sents]
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    scores = get_sentiment_analyzer().polarity_scores(cleaned)

    return {
        "char_count": len(cleaned),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "sentences": sentences,
        "sentiment_label": sentiment_label(scores["compound"]),
        "sentiment_compound": scores["compound"],
        "sentiment_scores": scores,
        "entities": extract_entities(doc),
    }
