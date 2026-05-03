# Integration 6A — Entity Analysis Pipeline

Module 6 Week A integration task for AI.SPIRE Applied AI & ML Systems.

Build a corpus-level entity analysis pipeline that processes climate articles, extracts entities, computes statistics, and produces visualizations.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Tasks

Complete the six functions in `entity_analysis.py`:
1. `load_corpus(filepath)` — Load the climate articles dataset
2. `preprocess_corpus(df)` — Add a language-aware `processed_text` column (NFC normalization; Arabic rows pass through without crashing; raw `text` preserved for NER)
3. `run_ner_pipeline(df, nlp)` — Filter to English and extract entities using the injected spaCy pipeline
4. `aggregate_entity_stats(entity_df, articles_df)` — Compute top entities, label counts, co-occurrence pairs, and per-category breakdown
5. `visualize_entity_distribution(stats, output_path)` — Create a bar chart of top entities
6. `generate_report(stats, co_occurrence)` — Produce a structured entity analysis report string

## Submission

1. Create a branch: `integration-6a-entity-analysis`
2. Complete `entity_analysis.py`
3. Open a PR to `main`
4. Paste your PR URL into TalentLMS → Module 6 Week A → Integration 6A

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted