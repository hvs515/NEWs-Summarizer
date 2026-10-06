import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from news_summarizer import analyze_article
from news_summarizer.config import get_api_key

SAMPLES = json.loads(
    (Path(__file__).resolve().parents[1] / "resources" / "sample_articles.json").read_text()
)

st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")
st.title("📰 NLP News Article Summarizer")
st.caption("Classical NLP (spaCy, VADER) + LLM summarization, with built-in quality checks.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key (optional)", type="password") or get_api_key()
    modes = ["llm", "extractive"] if api_key else ["extractive"]
    mode = st.radio("Summarizer", modes, help="Without an API key only the offline extractive mode is available.")
    sample = st.selectbox("Load a sample article", ["—"] + list(SAMPLES))

text = st.text_area(
    "Paste a news article",
    value=SAMPLES.get(sample, ""),
    height=250,
)

if st.button("Analyze", type="primary"):
    try:
        with st.spinner("Analyzing..."):
            result = analyze_article(text, mode=mode, api_key=api_key)
    except (ValueError, RuntimeError) as err:
        st.error(str(err))
        st.stop()

    nlp, out, ev = result["nlp"], result["summary"], result["evaluation"]

    st.subheader(out.get("headline", ""))
    st.write(out.get("summary", ""))
    st.markdown("**Key points**")
    for point in out.get("key_points", []):
        st.markdown(f"- {point}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", nlp["sentiment_label"], f"{nlp['sentiment_compound']:+.2f}")
    c2.metric("Words", nlp["word_count"])
    c3.metric("Compression", f"{ev['compression_ratio']:.0%}")
    c4.metric("Entity coverage", f"{ev['entity_coverage']:.0%}")

    if ev["unsupported_numbers"]:
        st.warning(f"Numbers in the summary not found in the article: {', '.join(ev['unsupported_numbers'])}")

    st.markdown("**Named entities**")
    st.dataframe(
        pd.DataFrame(
            [(cat, ", ".join(vals) or "—") for cat, vals in nlp["entities"].items()],
            columns=["Type", "Entities"],
        ),
        hide_index=True,
        use_container_width=True,
    )
