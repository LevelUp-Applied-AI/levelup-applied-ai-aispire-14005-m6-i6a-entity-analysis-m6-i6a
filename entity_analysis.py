"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Run:
python entity_analysis.py
"""

import unicodedata
from itertools import combinations
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import spacy


def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset."""
    df = pd.read_csv(filepath)
    return df


def preprocess_corpus(df):
    """Add a language-aware processed_text column."""
    corpus = df.copy()

    def normalize_text(text):
        if pd.isna(text):
            return ""
        return unicodedata.normalize("NFC", str(text))

    corpus["processed_text"] = corpus["text"].apply(normalize_text)

    return corpus


def run_ner_pipeline(df, nlp):
    """Run spaCy NER on English rows only."""
    english_df = df[df["language"] == "en"].copy()

    rows = []

    for _, row in english_df.iterrows():
        text_id = row["id"]
        text = row["text"]

        if pd.isna(text):
            continue

        doc = nlp(str(text))

        for ent in doc.ents:
            rows.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    entity_df = pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"]
    )

    return entity_df


def aggregate_entity_stats(entity_df, articles_df):
    """Compute entity statistics."""
    if entity_df.empty:
        return {
            "top_entities": pd.DataFrame(columns=["entity_text", "entity_label", "count"]),
            "label_counts": {},
            "co_occurrence": pd.DataFrame(columns=["entity_a", "entity_b", "co_count"]),
            "per_category": pd.DataFrame(columns=["category", "entity_label", "count"])
        }

    top_entities = (
        entity_df
        .groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )

    label_counts = entity_df["entity_label"].value_counts().to_dict()

    pair_counter = Counter()

    for text_id, group in entity_df.groupby("text_id"):
        unique_entities = sorted(set(group["entity_text"]))

        for entity_a, entity_b in combinations(unique_entities, 2):
            pair_counter[(entity_a, entity_b)] += 1

    co_rows = [
        {
            "entity_a": pair[0],
            "entity_b": pair[1],
            "co_count": count
        }
        for pair, count in pair_counter.items()
    ]

    co_occurrence = (
        pd.DataFrame(co_rows)
        .sort_values("co_count", ascending=False)
        .head(50)
        if co_rows
        else pd.DataFrame(columns=["entity_a", "entity_b", "co_count"])
    )

    articles_small = articles_df[["id", "category"]].copy()

    entity_with_category = entity_df.merge(
        articles_small,
        left_on="text_id",
        right_on="id",
        how="left"
    )

    per_category = (
        entity_with_category
        .groupby(["category", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["category", "count"], ascending=[True, False])
    )

    print("\nEntity statistics summary")
    print("-------------------------")
    print(f"Total extracted entities: {len(entity_df)}")
    print(f"Unique entities: {entity_df['entity_text'].nunique()}")
    print(f"Entity label counts: {label_counts}")
    print("\nTop entities:")
    print(top_entities.head(10))

    return {
        "top_entities": top_entities,
        "label_counts": label_counts,
        "co_occurrence": co_occurrence,
        "per_category": per_category
    }


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a horizontal bar chart of top entities."""
    top_entities = stats["top_entities"].copy()

    if top_entities.empty:
        print("No entities available to visualize.")
        return

    top_entities = top_entities.sort_values("count", ascending=True)

    labels = top_entities["entity_text"] + " (" + top_entities["entity_label"] + ")"

    plt.figure(figsize=(12, 8))
    plt.barh(labels, top_entities["count"])
    plt.xlabel("Frequency")
    plt.ylabel("Entity")
    plt.title("Top 20 Most Frequent Entities")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(stats, co_occurrence):
    """Generate a structured text report."""
    label_counts = stats["label_counts"]
    top_entities = stats["top_entities"].head(5)

    report_lines = []

    report_lines.append("Entity Analysis Report")
    report_lines.append("=" * 30)

    report_lines.append("\nEntity Counts by Type:")
    for label, count in label_counts.items():
        report_lines.append(f"- {label}: {count}")

    report_lines.append("\nTop 5 Most Frequent Entities:")
    for _, row in top_entities.iterrows():
        report_lines.append(
            f"- {row['entity_text']} ({row['entity_label']}): {row['count']}"
        )

    report_lines.append("\nTop 3 Co-occurring Entity Pairs:")
    if co_occurrence is not None and not co_occurrence.empty:
        for _, row in co_occurrence.head(3).iterrows():
            report_lines.append(
                f"- {row['entity_a']} + {row['entity_b']}: {row['co_count']} texts"
            )
    else:
        report_lines.append("- No co-occurring pairs found.")

    report_lines.append("\nSummary:")
    report_lines.append(
        "The entity analysis shows which people, organizations, locations, dates, "
        "and other named entities appear most often in the climate article corpus. "
        "Frequent entity labels help identify the main focus of the dataset, while "
        "co-occurring entity pairs show relationships between topics, places, and "
        "organizations. The per-category breakdown can help compare how policy, "
        "science, impact, and adaptation articles discuss climate issues differently."
    )

    return "\n".join(report_lines)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    raw = load_corpus()
    corpus = preprocess_corpus(raw)

    print(f"Corpus: {len(corpus)} articles")
    print(f"Languages: {corpus['language'].value_counts().to_dict()}")
    print(f"Categories: {corpus['category'].value_counts().to_dict()}")

    entities = run_ner_pipeline(corpus, nlp)
    print(f"\nExtracted {len(entities)} entities")

    stats = aggregate_entity_stats(entities, corpus)

    print(f"\nLabel counts: {stats['label_counts']}")
    print("\nTop 5 entities:")
    print(stats["top_entities"].head())

    print("\nPer-category counts:")
    print(stats["per_category"].head())

    visualize_entity_distribution(stats)
    print("\nVisualization saved to entity_distribution.png")

    report = generate_report(stats, stats["co_occurrence"])

    print("\n" + "=" * 50)
    print(report)