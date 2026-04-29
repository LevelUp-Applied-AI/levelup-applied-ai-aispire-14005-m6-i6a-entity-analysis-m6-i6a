"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import matplotlib

def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset."""
    return pd.read_csv(filepath)

def preprocess_corpus(df):
    """
    Add a language-aware `processed_text` column to the corpus.
    """
    # 1. Create a copy to prevent modifying the source dataframe
    df_copy = df.copy()

    def process_text(row):
        # Apply Unicode NFC normalization to handle multi-byte representations
        normalized = unicodedata.normalize('NFC', str(row['text']))
        
        # English rows are kept for NER, Arabic rows are passed to avoid pipeline crashes
        if row['language'] == 'en':
            return normalized
        elif row['language'] == 'ar':
            return normalized
        else:
            return ""

    # Apply the normalization while keeping original 'text' for NER signals
    df_copy['processed_text'] = df_copy.apply(process_text, axis=1)

    # --- Structured Output Logging ---
    print("\n" + "="*50)
    print("TASK 1: PREPROCESSING LOGS")
    print("="*50)
    
    # Show language counts
    lang_counts = df_copy['language'].value_counts().to_dict()
    print(f"Language Distribution: {lang_counts}")
    
    print("-" * 50)
    
    # Display a clean preview of the processed column
    print("Sample of 'processed_text' column:")
    for i, text in enumerate(df_copy['processed_text'].head(3), 1):
        preview = text[:75] + "..." if len(text) > 75 else text
        print(f"   {i}. {preview}")
    
    print("="*50)
    print("STATUS: Preprocessing complete.\n")
    
    return df_copy
def run_ner_pipeline(df, nlp):
    """
    Run spaCy NER on the English rows of a preprocessed corpus.
    """
    # Filter to English-language rows only
    en_df = df[df['language'] == 'en'].copy()
    
    entity_rows = []
    
    # Use nlp.pipe for optimized batch processing
    # Batch size of 20 is suitable for a CPU-native model like en_core_web_sm
    for doc, text_id in zip(nlp.pipe(en_df['text'], batch_size=20), en_df['id']):
        for ent in doc.ents:
            entity_rows.append({
                'text_id': text_id,
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
            
    # Return as a structured DataFrame
    return pd.DataFrame(entity_rows)

def aggregate_entity_stats(entity_df, articles_df):
    """Compute frequency, co-occurrence, and per-category statistics."""
    
    # 1. Top 20 most frequent entities
    top_entities = entity_df.groupby(['entity_text', 'entity_label']).size().reset_index(name='count')
    top_entities = top_entities.sort_values(by='count', ascending=False).head(20)
    
    # 2. Total count per entity label
    label_counts = entity_df['entity_label'].value_counts().to_dict()
    
    # 3. Co-occurrence: Count pairs appearing in the same text_id
    # We join the entity dataframe with itself on text_id
    pairs = entity_df.merge(entity_df, on='text_id')
    # Filter to keep only unique pairs (avoid self-comparison and double counting)
    pairs = pairs[pairs['entity_text_x'] < pairs['entity_text_y']]
    co_occurrence = pairs.groupby(['entity_text_x', 'entity_text_y']).size().reset_index(name='co_count')
    # Cap result at top 50 pairs by co_count
    co_occurrence = co_occurrence.sort_values(by='co_count', ascending=False).head(50)
    co_occurrence.columns = ['entity_a', 'entity_b', 'co_count']
    
    # 4. Per-category counts
    # Join entity_df with articles_df on text_id = id
    merged = entity_df.merge(articles_df[['id', 'category']], left_on='text_id', right_on='id')
    per_category = merged.groupby(['category', 'entity_label']).size().reset_index(name='count')
    
    return {
        'top_entities': top_entities,
        'label_counts': label_counts,
        'co_occurrence': co_occurrence,
        'per_category': per_category
    }

def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency."""
    top_df = stats['top_entities']
    
    # Create horizontal bar chart for better readability of entity names
    plt.figure(figsize=(12, 8))
    # Color bars by entity label using a color map
    unique_labels = top_df['entity_label'].unique()
    colors = matplotlib.colormaps.get_cmap('Set3').resampled(len(unique_labels)) 
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
    
    plt.barh(top_df['entity_text'], top_df['count'], color=[color_map[l] for l in top_df['entity_label']])
    
    plt.xlabel('Frequency (Count)')
    plt.ylabel('Entity Name')
    plt.title('Top 20 Entities by Frequency in Climate Corpus')
    plt.gca().invert_yaxis()  # Highest counts at the top
    
    # Add legend for entity types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[l], label=l) for l in unique_labels]
    plt.legend(handles=legend_elements, title="Entity Type", loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings."""
    label_summary = "\n".join([f"- {label}: {count}" for label, count in stats['label_counts'].items()])
    top_5 = stats['top_entities'].head(5)
    top_5_str = ", ".join([f"{row['entity_text']} ({row['entity_label']})" for _, row in top_5.iterrows()])
    
    top_3_co = co_occurrence.head(3)
    co_str = "\n".join([f"- {row['entity_a']} & {row['entity_b']} ({row['co_count']} texts)" for _, row in top_3_co.iterrows()])
    
    report = f"""
ENTITY ANALYSIS REPORT
======================
1. ENTITY COUNTS PER TYPE:
{label_summary}

2. TOP 5 MOST FREQUENT ENTITIES:
{top_5_str}

3. TOP 3 CO-OCCURRING PAIRS:
{co_str}

SUMMARY:
The analysis reveals a heavy emphasis on organizations and geographic locations related to global 
climate policy. The frequent co-occurrence of specific policy-related entities suggests a 
highly interconnected discourse within the climate dataset.
"""
    return report

if __name__ == "__main__":
    # Dependency injection pattern: load nlp once
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        print(f"Dataset loaded: {len(raw)} rows.")
        corpus = preprocess_corpus(raw)
        
        if corpus is not None:
            print(f"Preprocessing complete. Corpus size: {len(corpus)} articles.")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")

            # Run NER on English rows
            print("\nRunning NER pipeline (English articles only)...")
            entities = run_ner_pipeline(corpus, nlp)
            
            if entities is not None:
                print(f"Extraction complete: Found {len(entities)} entity mentions.")

                # Aggregate statistics
                print("Computing statistics...")
                stats = aggregate_entity_stats(entities, corpus)
                
                if stats is not None:
                    # Visualization
                    visualize_entity_distribution(stats)
                    print("Visualization saved to entity_distribution.png")

                    # Generate report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)