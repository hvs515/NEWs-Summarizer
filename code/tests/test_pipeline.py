import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from news_summarizer import analyze_article, format_report
from news_summarizer.classical import analyze_classical_nlp, clean_text, sentiment_label
from news_summarizer.evaluation import evaluate_summary
from news_summarizer.extractive import extractive_summary
from news_summarizer.llm import query_llm_summarizer, validate_article

SAMPLES = json.loads(
    (Path(__file__).resolve().parents[2] / "resources" / "sample_articles.json").read_text()
)


class FakeClient:
    """Stands in for openai.OpenAI so tests never hit the network."""

    def __init__(self, content):
        create = lambda **kwargs: SimpleNamespace(  # noqa: E731
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_clean_text_collapses_whitespace():
    assert clean_text("  a \n\n b\tc ") == "a b c"


@pytest.mark.parametrize("score,label", [(0.5, "Positive"), (-0.5, "Negative"), (0.0, "Neutral")])
def test_sentiment_label(score, label):
    assert sentiment_label(score) == label


def test_classical_nlp_on_business_sample():
    stats = analyze_classical_nlp(SAMPLES["business"])
    assert stats["sentence_count"] >= 4
    assert stats["word_count"] > 100
    assert "Sarah Jenkins" in stats["entities"]["People"]
    assert "Singapore" in stats["entities"]["Locations"]
    assert stats["sentiment_label"] == "Positive"


@pytest.mark.parametrize("name", list(SAMPLES))
def test_extractive_summary_shape(name):
    out = extractive_summary(SAMPLES[name])
    assert out["summary"] and out["headline"]
    assert 1 <= len(out["key_points"]) <= 5
    assert len(out["summary"].split()) <= 120
    # extractive sentences come straight from the source
    for point in out["key_points"]:
        assert point in clean_text(SAMPLES[name])


@pytest.mark.parametrize("bad", ["", "   ", "x" * 7000])
def test_validate_article_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_article(bad)


def test_llm_summarizer_parses_json():
    payload = {"summary": "s", "key_points": ["a"] * 5, "headline": "h"}
    out = query_llm_summarizer("Some article.", client=FakeClient(json.dumps(payload)))
    assert out == payload


def test_llm_summarizer_bad_json():
    with pytest.raises(RuntimeError):
        query_llm_summarizer("Some article.", client=FakeClient("not json"))


def test_evaluation_flags_unsupported_numbers():
    source = "Revenue rose to $14.2 billion, up 18%."
    ev = evaluate_summary(source, "Revenue rose 25% to $14.2 billion.", {"Dates": []})
    assert ev["unsupported_numbers"] == ["25"]


def test_pipeline_extractive_report():
    result = analyze_article(SAMPLES["technology"], mode="extractive")
    assert result["mode"] == "extractive"
    assert result["evaluation"]["unsupported_numbers"] == []
    report = format_report(result)
    assert "HEADLINE:" in report and "SUMMARY QUALITY CHECKS:" in report


def test_pipeline_llm_mode_with_fake_client():
    payload = {"summary": "Aura Technologies released the Aura-X9 chip.", "key_points": [], "headline": "h"}
    result = analyze_article(SAMPLES["technology"], mode="llm", client=FakeClient(json.dumps(payload)))
    assert result["mode"] == "llm"
    assert result["evaluation"]["entity_coverage"] > 0
