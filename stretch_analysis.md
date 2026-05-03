# Stretch 6A-S2 — Multilingual NER Comparison

## Label Schema Choice

For this comparison, I kept each model’s native label schema instead of mapping all labels into one shared schema. This makes the comparison clearer because both multilingual models use broad labels like PER, LOC, ORG, and MISC. Since the Arabic texts do not have gold-standard annotations, the Arabic evaluation is qualitative and based on entity counts, examples, entity density, and no-entity rate.

## Comparison Table

The comparison table is saved in:

`results/multilingual_ner_comparison.csv`

| Language | Model | Texts | Total Entities | Density / 100 Words | No Entity Rate | Entity Type Counts | Examples |
|---|---|---:|---:|---:|---:|---|---|
| English | spaCy_xx_ent_wiki_sm | 20 | 93 | 7.62 | 0.0% | MISC: 15, PER: 12, ORG: 39, LOC: 27 | IPCC, Sixth Assessment Report, Celsius |
| Arabic | spaCy_xx_ent_wiki_sm | 20 | 20 | 2.15 | 25.0% | PER: 9, MISC: 7, ORG: 2, LOC: 2 | وأكد التقرير, وقّع الأردن, وأكد وزير |
| English | HF_xlm_roberta_wikiann | 20 | 93 | 7.62 | 0.0% | PER: 10, ORG: 55, LOC: 28 | Antonio Guterres, COP, Dubai |
| Arabic | HF_xlm_roberta_wikiann | 20 | 64 | 6.88 | 0.0% | ORG: 33, LOC: 29, PER: 2 | الهيئة الحكومية الدولية المعنية بتغير المناخ, الأردن, البنك الدولي |

## Analysis

The English texts were handled more consistently by both models. Both spaCy and Hugging Face found 93 entities in the 20 English texts, with an entity density of 7.62 entities per 100 words and a 0% no-entity rate. Many English entities were clear climate-related organizations, locations, and people, such as `World Bank`, `NASA`, `European Union`, `Antonio Guterres`, `Dubai`, and `Middle East`. This makes sense because English named entities often have capitalization, clearer word boundaries, and more representation in multilingual training data.

Arabic performance was weaker and more uneven, especially for spaCy. The spaCy multilingual model only found 20 Arabic entities with a density of 2.15 entities per 100 words and a 25% no-entity rate. It also produced several weak examples such as `وأكد التقرير` labeled as PER and `وقّع الأردن` labeled as MISC, which are not clean named entities. Hugging Face performed better on Arabic, finding 64 entities with a density of 6.88 and a 0% no-entity rate. It identified useful Arabic examples like `الهيئة الحكومية الدولية المعنية بتغير المناخ`, `الأردن`, `البنك الدولي`, `دبي`, and `الأمم المتحدة`. However, it still made mistakes such as labeling `البيئة الأردن` as PER and climate terms like `ثاني أكسيد الكربون` as ORG. This shows that bilingual NLP systems in the MENA region need more than just a multilingual model. For real applications, Arabic outputs should be reviewed, tested on local data, and possibly improved with Arabic-specific fine-tuning.