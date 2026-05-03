import pandas as pd
import spacy
from transformers import pipeline
from collections import Counter, defaultdict
from pathlib import Path


DATA_PATH = "data/climate_articles.csv"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(filepath):
    df = pd.read_csv(filepath)

    # Normalize possible column names
    if "text" not in df.columns:
        possible_text_cols = ["article", "content", "body"]
        for col in possible_text_cols:
            if col in df.columns:
                df = df.rename(columns={col: "text"})
                break

    if "language" not in df.columns:
        possible_lang_cols = ["lang", "locale"]
        for col in possible_lang_cols:
            if col in df.columns:
                df = df.rename(columns={col: "language"})
                break

    if "text" not in df.columns or "language" not in df.columns:
        raise ValueError("Dataset must contain text and language columns.")

    df = df.dropna(subset=["text", "language"])
    df["language"] = df["language"].astype(str).str.lower()

    english_df = df[df["language"].isin(["english", "en"])].head(20)
    arabic_df = df[df["language"].isin(["arabic", "ar"])].head(20)

    return english_df, arabic_df


def run_spacy_model(texts):
    nlp = spacy.load("xx_ent_wiki_sm")
    rows = []

    for idx, text in enumerate(texts):
        doc = nlp(str(text))
        entities = []

        for ent in doc.ents:
            entities.append({
                "text_id": idx,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

        rows.extend(entities)

    return rows


def run_hf_model(texts):
    ner = pipeline(
        "token-classification",
        model="Davlan/xlm-roberta-base-wikiann-ner",
        aggregation_strategy="simple"
    )

    rows = []

    for idx, text in enumerate(texts):
        predictions = ner(str(text))
        for ent in predictions:
            label = ent.get("entity_group", ent.get("entity", ""))

            rows.append({
                "text_id": idx,
                "entity_text": ent["word"],
                "entity_label": label,
                "start_char": ent.get("start", None),
                "end_char": ent.get("end", None),
                "score": round(float(ent.get("score", 0)), 4),
            })

    return rows


def calculate_word_count(texts):
    return sum(len(str(text).split()) for text in texts)


def summarize_entities(rows, language, model_name, texts):
    total_entities = len(rows)
    word_count = calculate_word_count(texts)
    density = round((total_entities / word_count) * 100, 2) if word_count else 0

    label_counts = Counter(row["entity_label"] for row in rows)

    examples = []
    for row in rows[:3]:
        examples.append(f'{row["entity_text"]} ({row["entity_label"]})')

    text_ids_with_entities = set(row["text_id"] for row in rows)
    no_entity_rate = round(
        ((len(texts) - len(text_ids_with_entities)) / len(texts)) * 100,
        2
    ) if len(texts) else 0

    return {
        "language": language,
        "model": model_name,
        "texts_processed": len(texts),
        "total_entities": total_entities,
        "entity_density_per_100_words": density,
        "no_entities_rate_percent": no_entity_rate,
        "entity_type_counts": dict(label_counts),
        "example_entities": "; ".join(examples) if examples else "No entities found"
    }


def save_entity_outputs(rows, language, model_name):
    df = pd.DataFrame(rows)
    filename = OUTPUT_DIR / f"{language}_{model_name}_entities.csv"
    df.to_csv(filename, index=False)


def main():
    english_df, arabic_df = load_data(DATA_PATH)

    english_texts = english_df["text"].tolist()
    arabic_texts = arabic_df["text"].tolist()

    all_summaries = []

    experiments = [
        ("English", "spaCy_xx_ent_wiki_sm", english_texts, run_spacy_model),
        ("Arabic", "spaCy_xx_ent_wiki_sm", arabic_texts, run_spacy_model),
        ("English", "HF_xlm_roberta_wikiann", english_texts, run_hf_model),
        ("Arabic", "HF_xlm_roberta_wikiann", arabic_texts, run_hf_model),
    ]

    for language, model_name, texts, model_func in experiments:
        print(f"Running {model_name} on {language} texts...")
        rows = model_func(texts)

        save_entity_outputs(rows, language.lower(), model_name)
        summary = summarize_entities(rows, language, model_name, texts)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_DIR / "multilingual_ner_comparison.csv", index=False)

    print("\nMultilingual NER Comparison:")
    print(summary_df.to_markdown(index=False))

    print("\nSaved outputs to results/")


if __name__ == "__main__":
    main()