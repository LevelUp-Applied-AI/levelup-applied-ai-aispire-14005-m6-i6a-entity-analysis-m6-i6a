"""Autograder tests for Integration 6A — Entity Analysis Pipeline."""

import pytest
import pandas as pd
import numpy as np
import spacy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from entity_analysis import (load_corpus, preprocess_corpus, run_ner_pipeline,
                             aggregate_entity_stats,
                             visualize_entity_distribution, generate_report)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_articles.csv")

SAMPLE_NER_TEXT = (
    "The United Nations Environment Programme published a report on "
    "Jordan's climate adaptation in March 2024."
)


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def raw_corpus():
    data = load_corpus(DATA_PATH)
    assert data is not None, "load_corpus returned None"
    return data


@pytest.fixture
def corpus(raw_corpus):
    data = preprocess_corpus(raw_corpus)
    assert data is not None, "preprocess_corpus returned None"
    return data


def test_load_corpus(raw_corpus):
    """Corpus loads with expected columns and both languages present."""
    assert len(raw_corpus) > 0, "Corpus is empty"
    required_cols = {"id", "text", "source", "language", "category"}
    assert required_cols.issubset(set(raw_corpus.columns)), (
        f"Missing columns: {required_cols - set(raw_corpus.columns)}"
    )
    langs = set(raw_corpus["language"].unique())
    assert {"en", "ar"}.issubset(langs), (
        f"Expected both 'en' and 'ar' languages, got {langs}"
    )


def test_preprocess_corpus_adds_processed_text(corpus, raw_corpus):
    """preprocess_corpus preserves rows and adds a processed_text column."""
    assert len(corpus) == len(raw_corpus), (
        "preprocess_corpus should not drop rows"
    )
    assert "processed_text" in corpus.columns, (
        "Missing 'processed_text' column"
    )
    assert "text" in corpus.columns, (
        "Original 'text' column must be preserved (NER reads raw text)"
    )
    # processed_text must be strings for every row (no crashes on Arabic)
    for val in corpus["processed_text"]:
        assert isinstance(val, str), (
            f"processed_text must be a string for every row, got {type(val)}"
        )


def test_preprocess_corpus_nfc_normalized(corpus):
    """English processed_text is NFC-normalized."""
    import unicodedata
    en_rows = corpus[corpus["language"] == "en"]
    for val in en_rows["processed_text"]:
        if val:
            assert unicodedata.is_normalized("NFC", val), (
                "English processed_text must be NFC-normalized"
            )


def test_preprocess_corpus_handles_arabic(corpus):
    """Arabic rows survive preprocessing without error."""
    ar_rows = corpus[corpus["language"] == "ar"]
    if len(ar_rows) > 0:
        # Just verify the column exists and is a string — no-crash guarantee
        for val in ar_rows["processed_text"]:
            assert isinstance(val, str), "Arabic processed_text must be str"


def test_run_ner_pipeline_returns_dataframe(nlp):
    """run_ner_pipeline returns a DataFrame with required columns and filters to English."""
    sample_df = pd.DataFrame({
        "id": [1, 2],
        "text": [SAMPLE_NER_TEXT, "نص عربي لا ينبغي أن يُعالج"],
        "source": ["test", "test"],
        "language": ["en", "ar"],
        "category": ["policy", "policy"],
        "processed_text": [SAMPLE_NER_TEXT, ""],
    })
    result = run_ner_pipeline(sample_df, nlp)
    assert result is not None, "run_ner_pipeline returned None"
    assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
    required_cols = {"text_id", "entity_text", "entity_label"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )
    assert len(result) > 0, "No entities extracted from text with known entities"
    # Only the English row (id=1) should have produced entities
    assert set(result["text_id"].unique()).issubset({1}), (
        f"Non-English rows should be filtered. Got text_ids: {set(result['text_id'].unique())}"
    )


def test_aggregate_entity_stats():
    """aggregate_entity_stats returns frequency, co-occurrence, and per-category data."""
    entity_df = pd.DataFrame({
        "text_id": [1, 1, 1, 2, 2],
        "entity_text": ["IPCC", "Jordan", "March 2024", "IPCC", "Dead Sea"],
        "entity_label": ["ORG", "GPE", "DATE", "ORG", "LOC"],
    })
    articles_df = pd.DataFrame({
        "id": [1, 2],
        "text": ["text1", "text2"],
        "source": ["s", "s"],
        "language": ["en", "en"],
        "category": ["policy", "impact"],
    })
    result = aggregate_entity_stats(entity_df, articles_df)
    assert result is not None, "aggregate_entity_stats returned None"
    assert isinstance(result, dict), "Must return a dictionary"
    for key in ["top_entities", "label_counts", "co_occurrence", "per_category"]:
        assert key in result, f"Missing '{key}' key"

    # top_entities should be a DataFrame
    top = result["top_entities"]
    assert isinstance(top, pd.DataFrame), "top_entities must be a DataFrame"
    assert len(top) > 0, "top_entities is empty"

    # label_counts
    lc = result["label_counts"]
    assert isinstance(lc, dict), "label_counts must be a dict"
    assert lc.get("ORG", 0) == 2, f"Expected ORG count 2, got {lc.get('ORG')}"

    # co_occurrence
    co = result["co_occurrence"]
    assert isinstance(co, pd.DataFrame), "co_occurrence must be a DataFrame"

    # per_category: ORG should appear under both 'policy' (text 1) and 'impact' (text 2)
    pc = result["per_category"]
    assert isinstance(pc, pd.DataFrame), "per_category must be a DataFrame"
    required_pc_cols = {"category", "entity_label", "count"}
    assert required_pc_cols.issubset(set(pc.columns)), (
        f"per_category missing columns: {required_pc_cols - set(pc.columns)}"
    )
    policy_org = pc[(pc["category"] == "policy") & (pc["entity_label"] == "ORG")]
    assert len(policy_org) == 1 and int(policy_org["count"].iloc[0]) == 1, (
        f"Expected policy/ORG count 1, got {policy_org.to_dict('records')}"
    )


def test_visualize_entity_distribution(tmp_path):
    """visualize_entity_distribution creates an output image file."""
    stats = {
        "top_entities": pd.DataFrame({
            "entity_text": ["IPCC", "Jordan", "UN"],
            "entity_label": ["ORG", "GPE", "ORG"],
            "count": [10, 8, 5],
        }),
        "label_counts": {"ORG": 15, "GPE": 8},
    }
    output_path = str(tmp_path / "test_chart.png")
    visualize_entity_distribution(stats, output_path=output_path)
    assert os.path.exists(output_path), f"Chart not saved to {output_path}"
    assert os.path.getsize(output_path) > 0, "Chart file is empty"


def test_generate_report():
    """generate_report returns a non-empty string report."""
    stats = {
        "top_entities": pd.DataFrame({
            "entity_text": ["IPCC", "Jordan", "UN", "COP28", "Dead Sea"],
            "entity_label": ["ORG", "GPE", "ORG", "EVENT", "LOC"],
            "count": [10, 8, 5, 4, 3],
        }),
        "label_counts": {"ORG": 15, "GPE": 8, "EVENT": 4, "LOC": 3},
    }
    co_occurrence = pd.DataFrame({
        "entity_a": ["IPCC", "Jordan"],
        "entity_b": ["COP28", "Dead Sea"],
        "co_count": [3, 2],
    })
    result = generate_report(stats, co_occurrence)
    assert result is not None, "generate_report returned None"
    assert isinstance(result, str), "Must return a string"
    assert len(result) > 50, "Report is too short"
