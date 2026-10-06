"""LLM-based abstractive summarization through the OpenAI API."""

import json

from .config import CONFIG, get_api_key

SYSTEM_PROMPT_TEMPLATE = """
You are an expert news summarization system.

Your task is to analyze the provided news article and output a JSON response containing an accurate summary, {num_points} key points, and a suggested headline.

### Constraints:
1. The summary must be highly faithful to the source, concise (approximately {target_summary_len} words), and written in neutral, journalistic language.
2. Do NOT invent facts, statistics, or external claims. If information is not in the source text, do not infer or include it.
3. The "key_points" array MUST contain exactly {num_points} bullet points.
4. Output MUST be valid, parseable JSON conforming strictly to the schema below.

### JSON Schema:
{{
  "summary": "A concise, neutral summary of the article.",
  "key_points": ["Key point 1", "..."],
  "headline": "An engaging, accurate headline generated from the article."
}}
"""


def validate_article(article_text):
    if not article_text or not article_text.strip():
        raise ValueError("The provided article is empty.")
    if len(article_text) > CONFIG["MAX_ARTICLE_LENGTH_CHARS"]:
        raise ValueError(
            f"Article exceeds maximum configured length of "
            f"{CONFIG['MAX_ARTICLE_LENGTH_CHARS']} characters."
        )


def build_system_prompt():
    return SYSTEM_PROMPT_TEMPLATE.format(
        target_summary_len=CONFIG["SUMMARY_LENGTH_WORDS"],
        num_points=CONFIG["NUM_KEY_POINTS"],
    )


def query_llm_summarizer(article_text, api_key=None, client=None):
    validate_article(article_text)

    if client is None:
        api_key = api_key or get_api_key()
        if not api_key:
            raise ValueError("API key missing: set the OPENAI_API_KEY environment variable.")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=CONFIG["MODEL_NAME"],
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": f"ARTICLE:\n{article_text}"},
            ],
            temperature=CONFIG["TEMPERATURE"],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}") from e

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse output as valid JSON: {e}") from e
