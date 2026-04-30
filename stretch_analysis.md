# Multilingual NER Analysis Report

## Comparison Table
# Multilingual NER Comparison Table

| Language   | Model                  |   Total Entities |   Density (per 100 words) | Type Breakdown                                   | Examples                                                                             |
|:-----------|:-----------------------|-----------------:|--------------------------:|:-------------------------------------------------|:-------------------------------------------------------------------------------------|
| Arabic     | HF (xlm-roberta)       |               64 |                      6.88 | {'ORG': 33, 'GPE': 29, 'PERSON': 2}              | الهيئة الحكومية الدولية المعنية بتغير المناخ (ORG); الأردن (GPE); البنك الدولي (ORG) |
| Arabic     | spaCy (xx_ent_wiki_sm) |               20 |                      2.15 | {'PERSON': 9, 'MISC': 7, 'ORG': 2, 'GPE': 2}     | وأكد التقرير (PERSON); وقّع الأردن (MISC); وأكد وزير (PERSON)                        |
| English    | HF (xlm-roberta)       |               93 |                      7.62 | {'ORG': 55, 'GPE': 28, 'PERSON': 10}             | Antonio Guterres (PERSON); COP (ORG); COP (ORG)                                      |
| English    | spaCy (xx_ent_wiki_sm) |               93 |                      7.62 | {'ORG': 39, 'GPE': 27, 'MISC': 15, 'PERSON': 12} | IPCC (MISC); Sixth Assessment Report (MISC); Celsius (PERSON)                        |

### No Entities Found Rate
- Arabic: spaCy 25.0%, HF 0.0%
- English: spaCy 0.0%, HF 0.0%


## Qualitative Analysis

**A) Language-Specific Challenges:**
Based on the results, Arabic NER proved more challenging than English, specifically for the spaCy model. While the Hugging Face model (xlm-roberta) was robust, spaCy's `xx_ent_wiki_sm` had a 25.0% failure rate in Arabic texts (finding zero entities). A clear challenge observed is the morphological complexity; for instance, the entity "الأردن" was correctly identified, but when prefixes are attached or context changes, spaCy struggled. English results were identical across both models (93 entities found), showing that multilingual models still prioritize English structures over the nuances of Arabic's right-to-left and non-capitalized script.

**B) Implications for MENA Professional Contexts:**
In Jordan’s bilingual professional environment, the 25% failure rate of lightweight models like spaCy on Arabic text is a significant risk. For building real-world NLP systems, this data shows that Hugging Face transformers (like XLM-RoBERTa) are non-negotiable for accuracy, despite being slower. A bilingual pipeline must account for this performance gap by either using stronger transformer models or implementing custom Arabic rules to ensure that critical entities in Arabic climate reports are not missed, as missed entities lead to incomplete data analysis in regional environmental reporting.