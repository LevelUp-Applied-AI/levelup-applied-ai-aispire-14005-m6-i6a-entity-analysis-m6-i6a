"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from itertools import combinations


def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset."""
    df = pd.read_csv(filepath)
    return df


def preprocess_corpus(df):
    """Add processed_text column with Unicode NFC normalization."""
    corpus = df.copy()

    processed = []

    for _, row in corpus.iterrows():
        text = str(row["text"])
        lang = row["language"]

        normalized = unicodedata.normalize("NFC", text)

        if lang == "en":
            processed.append(normalized)
        elif lang == "ar":
            processed.append(normalized)   # or ""
        else:
            processed.append(normalized)

    corpus["processed_text"] = processed
    return corpus


def run_ner_pipeline(df, nlp):
    """Run spaCy NER on English rows only."""
    english_df = df[df["language"] == "en"]

    rows = []

    for _, row in english_df.iterrows():
        text_id = row["id"]
        text = row["text"]

        doc = nlp(text)

        for ent in doc.ents:
            rows.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    entity_df = pd.DataFrame(rows)
    return entity_df


def aggregate_entity_stats(entity_df, articles_df):
    """Compute all required statistics."""

    top_entities = (
        entity_df.groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )

    label_counts = entity_df["entity_label"].value_counts().to_dict()

    pair_counts = {}

    grouped = entity_df.groupby("text_id")["entity_text"].unique()

    for entities in grouped:
        entities = sorted(entities)

        for pair in combinations(entities, 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    co_rows = []

    for pair, count in pair_counts.items():
        if count >= 2:
            co_rows.append({
                "entity_a": pair[0],
                "entity_b": pair[1],
                "co_count": count
            })

    co_occurrence = pd.DataFrame(co_rows)

    if not co_occurrence.empty:
        co_occurrence = co_occurrence.sort_values(
            "co_count", ascending=False
        ).head(50)

    merged = entity_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id",
        right_on="id",
        how="left"
    )

    per_category = (
        merged.groupby(["category", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    stats = {
        "top_entities": top_entities,
        "label_counts": label_counts,
        "co_occurrence": co_occurrence,
        "per_category": per_category
    }

    return stats


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create bar chart for top entities."""
    top = stats["top_entities"]

    plt.figure(figsize=(12, 8))

    plt.barh(top["entity_text"], top["count"])

    plt.xlabel("Frequency")
    plt.ylabel("Entity")
    plt.title("Top 20 Most Frequent Entities")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(stats, co_occurrence):
    """Generate text report."""

    report = []
    report.append("ENTITY ANALYSIS REPORT")
    report.append("=" * 40)

    report.append("\n1. Entity Counts Per Type:")
    for label, count in stats["label_counts"].items():
        report.append(f"- {label}: {count}")

    report.append("\n2. Top 5 Most Frequent Entities:")
    top5 = stats["top_entities"].head(5)

    for _, row in top5.iterrows():
        report.append(
            f"- {row['entity_text']} ({row['entity_label']}): {row['count']}"
        )

    report.append("\n3. Top 3 Co-occurring Pairs:")

    if not co_occurrence.empty:
        top3 = co_occurrence.head(3)

        for _, row in top3.iterrows():
            report.append(
                f"- {row['entity_a']} + {row['entity_b']} ({row['co_count']})"
            )
    else:
        report.append("- No repeated co-occurring pairs found.")

    report.append("\n4. Summary:")
    report.append(
        "The corpus is dominated by recurring organizations, locations, "
        "and people connected to climate policy and global environmental "
        "issues. Frequent co-occurrences suggest strong relationships "
        "between countries, institutions, and climate topics."
    )

    return "\n".join(report)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    # Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            # Run NER on English rows
            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                # Aggregate statistics
                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    # Visualize
                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    # Generate report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)
