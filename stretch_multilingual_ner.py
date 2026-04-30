import pandas as pd
import spacy
from transformers import pipeline
import re
import os

# 1. Model Initialization
print("Loading multilingual models...")
# Download: python -m spacy download xx_ent_wiki_sm
nlp_spacy = spacy.load("xx_ent_wiki_sm")
# Hugging Face Multilingual Model
ner_hf = pipeline("ner", model="Davlan/xlm-roberta-base-wikiann-ner", aggregation_strategy="simple")

# 2. Label Mapping (Option B from image_c5d238.png)
def map_label(label):
    label = label.upper()
    mapping = {
        'PER': 'PERSON', 'PERSON': 'PERSON',
        'LOC': 'GPE', 'GPE': 'GPE',
        'ORG': 'ORG',
        'MISC': 'MISC'
    }
    return mapping.get(label, 'MISC')

# 3. Data Loading and Language Splitting
if not os.path.exists('data/climate_articles.csv'):
    raise FileNotFoundError("Dataset not found in data/climate_articles.csv")

df = pd.read_csv('data/climate_articles.csv')

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', str(text)))

# Select 20 samples per language as per constraints
arabic_df = df[df['text'].apply(is_arabic)].head(20).copy()
english_df = df[~df['text'].apply(is_arabic)].head(20).copy()

results = []

# 4. Processing Function
def process_texts(data_frame, lang_name):
    total_words = 0
    empty_spacy = 0
    empty_hf = 0
    
    for idx, row in data_frame.iterrows():
        text = str(row['text'])
        words = text.split()
        word_count = len(words)
        total_words += word_count
        
        # spaCy NER
        doc = nlp_spacy(text)
        if len(doc.ents) == 0: empty_spacy += 1
        for ent in doc.ents:
            results.append({
                'Language': lang_name,
                'Model': 'spaCy (xx_ent_wiki_sm)',
                'Type': map_label(ent.label_),
                'Entity': ent.text,
                'WordCount': word_count
            })
            
        # Hugging Face NER
        hf_ents = ner_hf(text)
        if len(hf_ents) == 0: empty_hf += 1
        for ent in hf_ents:
            results.append({
                'Language': lang_name,
                'Model': 'HF (xlm-roberta)',
                'Type': map_label(ent['entity_group']),
                'Entity': ent['word'],
                'WordCount': word_count
            })
            
    return total_words, empty_spacy, empty_hf

# Run Processing
print("Processing Arabic texts...")
ar_words, ar_empty_spacy, ar_empty_hf = process_texts(arabic_df, 'Arabic')

print("Processing English texts...")
en_words, en_empty_spacy, en_empty_hf = process_texts(english_df, 'English')

# 5. Summary and Comparison Table Generation
res_df = pd.DataFrame(results)

summary_data = []
for (lang, model), group in res_df.groupby(['Language', 'Model']):
    total_ents = len(group)
    word_base = ar_words if lang == 'Arabic' else en_words
    density = (total_ents / word_base) * 100
    
    # Get 3 examples
    examples = group.head(3).apply(lambda x: f"{x['Entity']} ({x['Type']})", axis=1).tolist()
    
    # Calculate counts per type
    type_counts = group['Type'].value_counts().to_dict()
    
    summary_data.append({
        'Language': lang,
        'Model': model,
        'Total Entities': total_ents,
        'Density (per 100 words)': round(density, 2),
        'Type Breakdown': type_counts,
        'Examples': "; ".join(examples)
    })

summary_df = pd.DataFrame(summary_data)

# Export to Markdown for the deliverable
with open('comparison_table.md', 'w', encoding='utf-8') as f:
    f.write("# Multilingual NER Comparison Table\n\n")
    f.write(summary_df.to_markdown(index=False))
    f.write(f"\n\n### No Entities Found Rate\n")
    f.write(f"- Arabic: spaCy {ar_empty_spacy/20*100}%, HF {ar_empty_hf/20*100}%\n")
    f.write(f"- English: spaCy {en_empty_spacy/20*100}%, HF {en_empty_hf/20*100}%\n")

print("\nSuccess! Results saved to comparison_table.md")