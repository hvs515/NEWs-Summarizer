from .classical import analyze_classical_nlp
from .config import get_api_key
from .evaluation import evaluate_summary
from .extractive import extractive_summary
from .llm import query_llm_summarizer, validate_article


def analyze_article(text, mode="auto", api_key=None, client=None):
    validate_article(text)
    if mode == "auto":
        mode = "llm" if (client or api_key or get_api_key()) else "extractive"

    nlp_stats = analyze_classical_nlp(text)
    if mode == "llm":
        summary = query_llm_summarizer(text, api_key=api_key, client=client)
    elif mode == "extractive":
        summary = extractive_summary(text)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    evaluation = evaluate_summary(text, summary.get("summary", ""), nlp_stats["entities"])
    return {"mode": mode, "nlp": nlp_stats, "summary": summary, "evaluation": evaluation}


def format_report(result):
    nlp, out, ev = result["nlp"], result["summary"], result["evaluation"]

    def join(values):
        return ", ".join(values) if values else "None detected"

    lines = [
        "-" * 50,
        f"NEWS ARTICLE ANALYSIS  (summarizer: {result['mode']})",
        "-" * 50,
        "",
        "HEADLINE:",
        out.get("headline", "N/A"),
        "",
        "SUMMARY:",
        out.get("summary", "N/A"),
        "",
        "KEY POINTS:",
    ]
    lines += [f"{i}. {pt}" for i, pt in enumerate(out.get("key_points", []), 1)]
    lines += [
        "",
        "SENTIMENT:",
        f"{nlp['sentiment_label']} (VADER compound {nlp['sentiment_compound']:.2f})",
        "",
        "NAMED ENTITIES:",
    ]
    for category, values in nlp["entities"].items():
        lines.append(f"  {category}: {join(values)}")
    lines += [
        "",
        "ARTICLE STATISTICS:",
        f"  Words: {nlp['word_count']}  Sentences: {nlp['sentence_count']}  "
        f"Characters: {nlp['char_count']}",
        "",
        "SUMMARY QUALITY CHECKS:",
        f"  Summary length: {ev['summary_words']} words (target ~{ev['target_words']})",
        f"  Compression ratio: {ev['compression_ratio']}",
        f"  Entity coverage: {ev['entity_coverage']:.0%}",
        f"  Unsupported numbers: {join(ev['unsupported_numbers'])}",
        "-" * 50,
    ]
    return "\n".join(lines)
