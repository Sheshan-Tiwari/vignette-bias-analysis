# LLM Trauma Disclosure Bias - Automated Content Analysis

Automated content analysis of LLM responses to gendered trauma disclosures using three complementary methods:

- Semantic similarity using sentence embeddings (all-MiniLM-L6-v2)
- Sentiment analysis using RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- Response length analysis

Follows the counterfactual vignette methodology of Kim et al. (2023, JAMA Network Open) and Bouguettaya et al. (2025, npj Digital Medicine).

## Requirements
pip install sentence-transformers transformers torch pandas

## Usage
Update the DATA_PATH and OUTPUT_DIR variables in the script before running.
python3 analyze_responses_v2.py

## Note
Data not included due to sensitive content.
