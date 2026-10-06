# NLP-based News Article Summarizer

[![tests](https://github.com/hvs515/News-summarizer/actions/workflows/tests.yml/badge.svg)](https://github.com/hvs515/News-summarizer/actions/workflows/tests.yml)
<a href="https://colab.research.google.com/github/hvs515/News-summarizer/blob/main/notebooks/news.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Capstone project for the **Batch F NLP training program**. The system takes a raw news article and returns a headline, a summary, five key points, sentiment, named entities, text statistics, and automatic checks on the quality of the summary.

## Student details

| Field | Value |
|---|---|
| Name | _<your full name>_ |
| Registration Number | _<your reg. no., e.g. 229301XXX>_ |
| Branch | _<your branch>_ |
| Batch | F |
| Project Title | NLP-based News Article Summarizer |
| GitHub Username | [hvs515](https://github.com/hvs515) |
| Training Program | _<program name>_ – NLP Capstone |
| Capstone (team) repository | _<link to team capstone repo>_ |

## Features

- **Classical NLP** with spaCy and NLTK: text cleaning, sentence and word statistics, VADER sentiment, and named-entity recognition (people, organisations, locations, dates).
- **Abstractive summarization** with an LLM (OpenAI `gpt-4o-mini`). The prompt asks for strict JSON and tells the model not to invent facts.
- **Offline extractive summarizer** that works without an API key. It ranks sentences by word frequency, entities and position in the article. It is also used as the baseline.
- **Summary quality checks** that need no reference summary: compression ratio, entity coverage, unigram overlap, and a check that flags numbers in the summary which do not appear in the source.
- **Three ways to use it:** a Streamlit web app, a command-line tool, and a Colab notebook.
- **Automated tests** (pytest) run on every push with GitHub Actions.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── assignments/        # weekly training assignments
├── notebooks/          # Colab notebook (original exploration)
│   └── news.ipynb
├── code/
│   ├── app.py          # Streamlit web app
│   ├── news_summarizer/
│   │   ├── config.py       # settings + API key lookup
│   │   ├── classical.py    # spaCy / VADER analysis
│   │   ├── extractive.py   # offline extractive summarizer
│   │   ├── llm.py          # OpenAI summarizer + prompt
│   │   ├── evaluation.py   # reference-free quality checks
│   │   ├── pipeline.py     # end-to-end pipeline + text report
│   │   └── __main__.py     # CLI
│   └── tests/
├── resources/          # sample articles, references
├── presentations/      # slides for the final demo
└── capstone/           # project report, results, screenshots
```

## Installation guide

Requirements: Python 3.10 or newer.

```bash
git clone https://github.com/hvs515/News-summarizer.git
cd News-summarizer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # or en_core_web_md for better NER
```

Optional: to use the LLM summarizer, set your OpenAI key. Without a key, the project uses the offline extractive summarizer.

```bash
export OPENAI_API_KEY="sk-..."      # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

## Usage

**Web app**

```bash
streamlit run code/app.py
```

**Command line** (run from the `code/` folder)

```bash
cd code
python -m news_summarizer --sample technology
python -m news_summarizer --file my_article.txt --mode llm
python -m news_summarizer --all-samples --mode extractive --json-out results.json
```

**Notebook:** open `notebooks/news.ipynb` in Colab and add `OPENAI_API_KEY` under Colab Secrets.

**Tests**

```bash
python -m pytest -q
```

## Results

The sample outputs are in [`capstone/results/`](capstone/results/), and the screenshots are in [`capstone/screenshots/`](capstone/screenshots/). These are the results of the extractive baseline on the three bundled articles:

| Article | Summary words | Compression | Entity coverage | Unsupported numbers |
|---|---|---|---|---|
| Politics | 73 | 0.55 | 90% | none |
| Technology | 83 | 0.67 | 89% | none |
| Business | 71 | 0.56 | 80% | none |

![App screenshot](capstone/screenshots/02_analysis_technology.png)

See the [project report](capstone/README.md) for the method, the evaluation and the limitations.

## Contributing

We work on feature branches and merge through reviewed pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md).
