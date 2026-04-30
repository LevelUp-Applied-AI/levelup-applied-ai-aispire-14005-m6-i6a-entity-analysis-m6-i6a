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
