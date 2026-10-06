# Capstone Report: NLP-based News Article Summarizer

## 1. Problem statement

Readers and analysts deal with far more news than they can read. This project builds a system that turns a raw news article into a short, faithful summary, a headline and key points. It also returns structured information (sentiment, entities, statistics), so the user can understand the article at a glance.

## 2. Approach

```
article text
   │
   ├─► Classical NLP (spaCy + NLTK VADER)
   │      cleaning · sentences/words · sentiment · NER
   │
   ├─► Summarizer
   │      ├─ LLM (OpenAI, JSON-constrained prompt)   ← if API key
   │      └─ Extractive baseline (frequency + entity + position scoring)
   │
   └─► Evaluation (reference-free)
          compression · entity coverage · unigram overlap · unsupported numbers
                                   │
                                   ▼
                    report (CLI) / dashboard (Streamlit)
```

| Component | Technique | Why |
|---|---|---|
| Cleaning and statistics | Regex, spaCy tokenizer and sentence splitter | Deterministic and fast |
| Sentiment | NLTK VADER (compound score, threshold ±0.05) | Lexicon-based, so the score can be explained |
| NER | spaCy `en_core_web_md` (falls back to `sm`) | Pretrained, works on general news text |
| Abstractive summary | `gpt-4o-mini`, temperature 0.2, `response_format=json_object` | Fluent summaries, output that can be parsed |
| Extractive summary | Content-word frequency, plus an entity bonus and a lead-position bonus | Works offline, cannot hallucinate, and serves as the baseline |
| Headline (offline) | Lead sentence with the first appositive removed | News puts the main fact in the first sentence |

### Prompt design

The system prompt sets a target length of about 80 words and asks for exactly five key points and a JSON schema. It also says explicitly: *"Do NOT invent facts, statistics, or external claims."* JSON mode guarantees that the output can be parsed. If parsing fails, the code raises an error instead of returning partial output.

## 3. Evaluation

The project uses no gold-standard reference summaries, so the hand-written "Excellent / High" labels from the original notebook were replaced with **measurable, reference-free checks**:

- **Compression ratio**: summary words divided by article words.
- **Entity coverage**: the share of spaCy entities from the source that appear in the summary. This is a proxy for coverage of who, where and when.
- **Unsupported numbers**: figures that appear in the summary but not in the source. Numbers are the most damaging thing to hallucinate in news, and this check catches them cheaply.
- **Unigram precision and recall** against the source, a ROUGE-1-style overlap.

Results for the extractive baseline (`results/extractive_results.json`):

| Article | Summary words | Compression | Entity coverage | Unsupported numbers |
|---|---|---|---|---|
| Politics | 73 | 0.55 | 90% | none |
| Technology | 83 | 0.67 | 89% | none |
| Business | 71 | 0.56 | 80% | none |

To compare against the LLM, set `OPENAI_API_KEY` and run:

```bash
cd code && python -m news_summarizer --all-samples --mode llm --json-out ../capstone/results/llm_results.json
```

### Limitations of automatic metrics
- Overlap metrics such as ROUGE and BLEU measure shared words, not meaning. A correct paraphrase can score low.
- One negation ("not") can reverse the meaning of a sentence while overlap stays high.
- Entity coverage depends on NER quality. The small spaCy model sometimes tags words such as "annual" as dates.

## 4. Testing

`code/tests/test_pipeline.py` has 16 pytest tests. They cover cleaning, sentiment thresholds, NER on known entities, the shape of the extractive output, input validation (empty or too long), parsing of LLM JSON with a mocked client, and detection of unsupported numbers. GitHub Actions runs the tests on every push and pull request.

## 5. Deployment

- **Local web app:** `streamlit run code/app.py`
- **Streamlit Community Cloud:** point it at `code/app.py`, then add `OPENAI_API_KEY` under *Secrets* (optional).
- **CLI** for batch processing: `python -m news_summarizer --all-samples --json-out out.json`

## 6. Conclusion and future work

The hybrid design uses each technique where it works best. Classical NLP gives transparent, deterministic analysis, and the LLM gives fluent abstractive summaries. The extractive mode keeps the system usable when no API key is available. The quality checks give an objective signal for detecting hallucinations.

Future work:
- Evaluate on a labelled dataset such as CNN/DailyMail with ROUGE and BERTScore.
- Run an entailment model (NLI) to check whether each summary sentence is supported by the source.
- Accept multilingual input and fetch articles from a URL.
- Add a feedback step where a human reviews flagged summaries.

## 7. Team contributions

| Member | Role / contributions |
|---|---|
| _<name>_ | _<e.g. model development, prompt design>_ |
| _<name>_ | _<e.g. testing, documentation>_ |
| _<name>_ | _<e.g. frontend, deployment>_ |
