import os

CONFIG = {
    "MODEL_NAME": os.getenv("NEWS_SUMMARIZER_MODEL", "gpt-4o-mini"),
    "MAX_ARTICLE_LENGTH_CHARS": 6000,
    "SUMMARY_LENGTH_WORDS": 80,
    "TEMPERATURE": 0.2,
    "NUM_KEY_POINTS": 5,
    "SPACY_MODELS": ["en_core_web_md", "en_core_web_sm"],
}


def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata

        return userdata.get("OPENAI_API_KEY")
    except Exception:
        return None
