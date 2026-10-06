"""Command-line interface.

Examples:
    python -m news_summarizer --file article.txt
    python -m news_summarizer --sample technology --mode extractive
    python -m news_summarizer --all-samples --json-out results.json
"""

import argparse
import json
import sys
from pathlib import Path

from .pipeline import analyze_article, format_report

SAMPLES = Path(__file__).resolve().parents[2] / "resources" / "sample_articles.json"


def load_samples():
    return json.loads(SAMPLES.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize and analyze news articles.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Path to a .txt file with the article")
    src.add_argument("--text", help="Article text passed directly")
    src.add_argument("--sample", help="Name of a bundled sample article")
    src.add_argument("--all-samples", action="store_true", help="Run every bundled sample")
    parser.add_argument("--mode", choices=["auto", "llm", "extractive"], default="auto")
    parser.add_argument("--json-out", help="Also write the raw results as JSON")
    args = parser.parse_args(argv)

    if args.all_samples:
        articles = load_samples()
    elif args.sample:
        samples = load_samples()
        if args.sample not in samples:
            parser.error(f"unknown sample; choose from {', '.join(samples)}")
        articles = {args.sample: samples[args.sample]}
    elif args.file:
        articles = {Path(args.file).stem: Path(args.file).read_text(encoding="utf-8")}
    else:
        articles = {"input": args.text}

    results = {}
    for name, text in articles.items():
        try:
            result = analyze_article(text, mode=args.mode)
        except (ValueError, RuntimeError) as err:
            print(f"[{name}] error: {err}", file=sys.stderr)
            continue
        print(f"\n### {name}")
        print(format_report(result))
        result["nlp"].pop("sentences", None)
        results[name] = result

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
