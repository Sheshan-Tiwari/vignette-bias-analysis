"""
Automated Analysis of LLM Responses to Gendered Trauma Disclosures
===================================================================
Study: Gender-Based Differential in AI Trauma Disclosure Response
Method: Counterfactual vignette design across two LLMs (ChatGPT, Claude)

This script implements three complementary automated content analysis methods:

1. Semantic Similarity Analysis
   Measures how much each gendered response (Female/Male) deviates semantically
   from its matched neutral baseline using sentence embeddings and cosine distance.
   Follows Du et al. (2025, arXiv:2511.08225) and Bouguettaya et al. (2025,
   npj Digital Medicine).

2. Sentiment Analysis
   Measures the emotional tone (positive/negative/neutral) of each response
   relative to its neutral baseline. Captures directional differences in
   emotional warmth across gender conditions.

3. Response Length Analysis
   Measures word count per response across conditions. Length serves as a
   proxy for response elaboration — a behavioral signal grounded in WHO LIVES
   guidance on sustained emotional presence in trauma disclosure response.

Input:
    responses.csv — exported from Raw_Data tab of study spreadsheet
    Expected columns: Model, Condition, Run_Number, Turn1_Response

Output:
    similarity_results.csv — per-comparison cosine distances
    sentiment_results.csv  — per-response sentiment scores
    length_results.csv     — per-response word counts
    summary_results.csv    — combined summary table

References:
    Du, Y., Borchers, C., & Cukurova, M. (2025). Benchmarking educational LLMs
        with analytics: A case study on gender bias in feedback. arXiv:2511.08225.
    Bouguettaya, A., et al. (2025). [Racial bias in LLM clinical responses].
        npj Digital Medicine.
    Kim, S., et al. (2023). Physician AI. JAMA Network Open.
    WHO. (2014). Health care for women subjected to intimate partner violence
        or sexual violence: A clinical handbook (WHO/RHR/14.26).
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# Update DATA_PATH to point to your responses CSV file.
# Update OUTPUT_DIR to the folder where results should be saved.
# =============================================================================

DATA_PATH  = "PATH"
OUTPUT_DIR = "PATH"

# Models and conditions defined in study design
MODELS     = ["ChatGPT", "Claude"]
GENDERED   = ["Female", "Male"]
N_RUNS     = 5


# =============================================================================
# STEP 1 — LOAD AND VALIDATE DATA
# =============================================================================

def load_data(path):
    """
    Load response CSV and validate expected columns are present.
    Skips the first row which contains the sheet title rather than headers.
    Returns a cleaned DataFrame.
    """
    print("Loading data...")

    df = pd.read_csv(path, skiprows=1)

    # Strip whitespace from column names and key columns
    df.columns = df.columns.str.strip()
    df["Model"]          = df["Model"].str.strip()
    df["Condition"]      = df["Condition"].str.strip()
    df["Run_Number"]     = df["Run_Number"].astype(int)
    df["Turn1_Response"] = df["Turn1_Response"].astype(str).str.strip()

    # Validate expected structure
    required_cols = ["Model", "Condition", "Run_Number", "Turn1_Response"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  Loaded {len(df)} rows.")
    print(f"  Models found:     {list(df['Model'].unique())}")
    print(f"  Conditions found: {list(df['Condition'].unique())}")

    return df


# =============================================================================
# STEP 2 — SEMANTIC SIMILARITY ANALYSIS
# =============================================================================

def run_semantic_similarity(df, model):
    """
    For each gendered response (Female/Male), calculates cosine distance from
    its matched neutral baseline (same model, same run number).

    Cosine distance = 1 - cosine similarity.
    Distance of 0 = identical meaning. Larger distance = greater semantic shift.

    Returns a DataFrame with per-comparison cosine distances.
    """
    print("\nEmbedding all responses for semantic similarity analysis...")

    # Generate sentence embeddings for all responses
    responses  = df["Turn1_Response"].tolist()
    embeddings = model.encode(responses, show_progress_bar=True)
    df = df.copy()
    df["embedding"] = list(embeddings)

    print("  Embeddings complete.")
    print("\nCalculating cosine distances...")

    results = []

    for model_name in MODELS:
        for condition in GENDERED:
            for run_num in range(1, N_RUNS + 1):

                # Retrieve neutral baseline for this model and run
                neutral_row = df[
                    (df["Model"] == model_name) &
                    (df["Condition"] == "Neutral") &
                    (df["Run_Number"] == run_num)
                ]

                # Retrieve gendered response for this model and run
                gend_row = df[
                    (df["Model"] == model_name) &
                    (df["Condition"] == condition) &
                    (df["Run_Number"] == run_num)
                ]

                # Skip if either row is missing — log warning
                if neutral_row.empty or gend_row.empty:
                    print(f"  WARNING: Missing data — {model_name} {condition} Run {run_num}")
                    continue

                # Extract embedding vectors
                neutral_emb = neutral_row["embedding"].values[0].reshape(1, -1)
                gend_emb    = gend_row["embedding"].values[0].reshape(1, -1)

                # Calculate cosine similarity and convert to distance
                sim      = cosine_similarity(neutral_emb, gend_emb)[0][0]
                distance = 1 - sim

                results.append({
                    "Model":              model_name,
                    "Condition":          condition,
                    "Run":                run_num,
                    "Neutral_Reference":  f"{model_name}_Neutral_{run_num}",
                    "Comparison":         f"{model_name}_{condition}_{run_num} vs {model_name}_Neutral_{run_num}",
                    "Cosine_Similarity":  round(float(sim), 4),
                    "Cosine_Distance":    round(float(distance), 4),
                })

    return pd.DataFrame(results)


def print_similarity_summary(results_df):
    """Print per-comparison and summary similarity results to console."""

    print("\n" + "=" * 68)
    print("SEMANTIC SIMILARITY — PER-COMPARISON COSINE DISTANCES")
    print("=" * 68)
    print(f"{'Comparison':<48} {'Similarity':>10} {'Distance':>10}")
    print("-" * 68)
    for _, row in results_df.iterrows():
        print(f"{row['Comparison']:<48} {row['Cosine_Similarity']:>10.4f} "
              f"{row['Cosine_Distance']:>10.4f}")

    print("\n" + "=" * 68)
    print("SEMANTIC SIMILARITY — MEAN DISTANCE BY MODEL AND CONDITION")
    print("=" * 68)
    summary = results_df.groupby(["Model", "Condition"]).agg(
        Mean_Distance   = ("Cosine_Distance",   "mean"),
        SD_Distance     = ("Cosine_Distance",   "std"),
        Mean_Similarity = ("Cosine_Similarity", "mean"),
    ).round(4).reset_index()
    print(summary.to_string(index=False))

    print("\n" + "=" * 68)
    print("SEMANTIC SIMILARITY — KEY FINDING: FEMALE vs MALE")
    print("=" * 68)
    for model_name in MODELS:
        female_mean = results_df[
            (results_df["Model"] == model_name) &
            (results_df["Condition"] == "Female")
        ]["Cosine_Distance"].mean()

        male_mean = results_df[
            (results_df["Model"] == model_name) &
            (results_df["Condition"] == "Male")
        ]["Cosine_Distance"].mean()

        diff      = round(female_mean - male_mean, 4)
        direction = "Female deviates MORE from neutral" if diff > 0 \
                    else "Male deviates MORE from neutral"

        print(f"\n{model_name}:")
        print(f"  Female mean distance: {female_mean:.4f}")
        print(f"  Male mean distance:   {male_mean:.4f}")
        print(f"  Difference:           {diff:.4f} ({direction})")


# =============================================================================
# STEP 3 — SENTIMENT ANALYSIS
# =============================================================================

def run_sentiment_analysis(df):
    """
    Applies a pre-trained sentiment analysis model to each response.
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest — a RoBERTa model
    fine-tuned for sentiment classification producing Negative/Neutral/Positive
    labels with associated confidence scores.

    For each gendered response, also calculates the sentiment shift relative
    to its matched neutral baseline — the change in positive sentiment score
    from neutral to gendered condition.

    Returns a DataFrame with per-response sentiment scores and shifts.
    """
    print("\n" + "=" * 68)
    print("SENTIMENT ANALYSIS")
    print("=" * 68)
    print("Loading sentiment model (cardiffnlp/twitter-roberta-base-sentiment)...")

    # Load pre-trained sentiment pipeline
    # truncation=True handles responses longer than model's max token length
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
    )

    print("  Sentiment model loaded.")
    print("  Scoring all responses...")

    results = []

    for _, row in df.iterrows():
        # Run sentiment classification on response text
        output = sentiment_pipeline(row["Turn1_Response"])[0]

        results.append({
            "Model":          row["Model"],
            "Condition":      row["Condition"],
            "Run_Number":     row["Run_Number"],
            "Sentiment_Label":row["label"] if "label" in row else output["label"],
            "Sentiment_Score": round(output["score"], 4),
            "Raw_Label":      output["label"],
        })

    sentiment_df = pd.DataFrame(results)

    # Fix label column
    sentiment_df["Sentiment_Label"] = sentiment_df["Raw_Label"]
    sentiment_df = sentiment_df.drop(columns=["Raw_Label"])

    # Calculate sentiment shift for gendered responses relative to neutral baseline
    shift_results = []

    for model_name in MODELS:
        for condition in GENDERED:
            for run_num in range(1, N_RUNS + 1):

                neutral_sent = sentiment_df[
                    (sentiment_df["Model"] == model_name) &
                    (sentiment_df["Condition"] == "Neutral") &
                    (sentiment_df["Run_Number"] == run_num)
                ]

                gend_sent = sentiment_df[
                    (sentiment_df["Model"] == model_name) &
                    (sentiment_df["Condition"] == condition) &
                    (sentiment_df["Run_Number"] == run_num)
                ]

                if neutral_sent.empty or gend_sent.empty:
                    continue

                # Calculate shift in sentiment score from neutral to gendered
                # Positive shift = gendered response more positive than neutral
                # Negative shift = gendered response more negative than neutral
                neutral_score = neutral_sent["Sentiment_Score"].values[0]
                gend_score    = gend_sent["Sentiment_Score"].values[0]
                neutral_label = neutral_sent["Sentiment_Label"].values[0]
                gend_label    = gend_sent["Sentiment_Label"].values[0]

                # Assign sign based on label direction
                def signed_score(label, score):
                    if "positive" in label.lower():
                        return score
                    elif "negative" in label.lower():
                        return -score
                    else:
                        return 0

                shift = round(
                    signed_score(gend_label, gend_score) -
                    signed_score(neutral_label, neutral_score), 4
                )

                shift_results.append({
                    "Model":               model_name,
                    "Condition":           condition,
                    "Run":                 run_num,
                    "Neutral_Label":       neutral_label,
                    "Neutral_Score":       round(neutral_score, 4),
                    "Gendered_Label":      gend_label,
                    "Gendered_Score":      round(gend_score, 4),
                    "Sentiment_Shift":     shift,
                    "Shift_Direction":     "More positive" if shift > 0
                                          else "More negative" if shift < 0
                                          else "No change",
                })

    shift_df = pd.DataFrame(shift_results)

    # Print summary
    print("\n" + "=" * 68)
    print("SENTIMENT — MEAN SHIFT FROM NEUTRAL BY MODEL AND CONDITION")
    print("=" * 68)
    summary = shift_df.groupby(["Model", "Condition"]).agg(
        Mean_Sentiment_Shift = ("Sentiment_Shift", "mean"),
        SD_Shift             = ("Sentiment_Shift", "std"),
    ).round(4).reset_index()
    print(summary.to_string(index=False))

    print("\n" + "=" * 68)
    print("SENTIMENT — KEY FINDING: FEMALE vs MALE SHIFT FROM NEUTRAL")
    print("=" * 68)
    for model_name in MODELS:
        female_shift = shift_df[
            (shift_df["Model"] == model_name) &
            (shift_df["Condition"] == "Female")
        ]["Sentiment_Shift"].mean()

        male_shift = shift_df[
            (shift_df["Model"] == model_name) &
            (shift_df["Condition"] == "Male")
        ]["Sentiment_Shift"].mean()

        diff      = round(female_shift - male_shift, 4)
        direction = "Female more positive than Male" if diff > 0 \
                    else "Male more positive than Female"

        print(f"\n{model_name}:")
        print(f"  Female mean sentiment shift: {female_shift:.4f}")
        print(f"  Male mean sentiment shift:   {male_shift:.4f}")
        print(f"  Difference:                  {diff:.4f} ({direction})")

    return shift_df


# =============================================================================
# STEP 4 — RESPONSE LENGTH ANALYSIS
# =============================================================================

def run_length_analysis(df):
    """
    Calculates word count for each response.

    Response length serves as a behavioral proxy for elaboration — a signal
    grounded in WHO LIVES guidance that trauma-informed responders should
    sustain emotional presence rather than provide brief acknowledgments.

    For each gendered response, also calculates word count difference relative
    to its matched neutral baseline.

    Returns a DataFrame with per-response word counts and length differences.
    """
    print("\n" + "=" * 68)
    print("RESPONSE LENGTH ANALYSIS")
    print("=" * 68)

    # Calculate word count for every response
    df = df.copy()
    df["Word_Count"] = df["Turn1_Response"].apply(lambda x: len(str(x).split()))

    results = []

    for model_name in MODELS:
        for condition in GENDERED:
            for run_num in range(1, N_RUNS + 1):

                neutral_row = df[
                    (df["Model"] == model_name) &
                    (df["Condition"] == "Neutral") &
                    (df["Run_Number"] == run_num)
                ]

                gend_row = df[
                    (df["Model"] == model_name) &
                    (df["Condition"] == condition) &
                    (df["Run_Number"] == run_num)
                ]

                if neutral_row.empty or gend_row.empty:
                    continue

                neutral_wc = neutral_row["Word_Count"].values[0]
                gend_wc    = gend_row["Word_Count"].values[0]
                difference = gend_wc - neutral_wc

                results.append({
                    "Model":            model_name,
                    "Condition":        condition,
                    "Run":              run_num,
                    "Neutral_Words":    neutral_wc,
                    "Gendered_Words":   gend_wc,
                    "Word_Difference":  difference,
                    "Direction":        "Longer than neutral" if difference > 0
                                        else "Shorter than neutral" if difference < 0
                                        else "Same length",
                })

    length_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 68)
    print("LENGTH — MEAN WORD COUNT DIFFERENCE FROM NEUTRAL")
    print("=" * 68)
    summary = length_df.groupby(["Model", "Condition"]).agg(
        Mean_Gendered_Words  = ("Gendered_Words",  "mean"),
        Mean_Neutral_Words   = ("Neutral_Words",   "mean"),
        Mean_Word_Difference = ("Word_Difference", "mean"),
        SD_Word_Difference   = ("Word_Difference", "std"),
    ).round(2).reset_index()
    print(summary.to_string(index=False))

    print("\n" + "=" * 68)
    print("LENGTH — KEY FINDING: FEMALE vs MALE WORD DIFFERENCE FROM NEUTRAL")
    print("=" * 68)
    for model_name in MODELS:
        female_diff = length_df[
            (length_df["Model"] == model_name) &
            (length_df["Condition"] == "Female")
        ]["Word_Difference"].mean()

        male_diff = length_df[
            (length_df["Model"] == model_name) &
            (length_df["Condition"] == "Male")
        ]["Word_Difference"].mean()

        diff      = round(female_diff - male_diff, 2)
        direction = "Female responses longer than Male" if diff > 0 \
                    else "Male responses longer than Female"

        print(f"\n{model_name}:")
        print(f"  Female mean word difference from neutral: {female_diff:.2f}")
        print(f"  Male mean word difference from neutral:   {male_diff:.2f}")
        print(f"  Difference:                               {diff:.2f} ({direction})")

    return length_df


# =============================================================================
# STEP 5 — SAVE ALL RESULTS
# =============================================================================

def save_results(similarity_df, sentiment_df, length_df, output_dir):
    """
    Save all analysis results to CSV files in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save similarity results
    sim_path = os.path.join(output_dir, "similarity_results.csv")
    similarity_df[["Model","Condition","Run","Neutral_Reference",
                   "Cosine_Similarity","Cosine_Distance","Comparison"]].to_csv(
        sim_path, index=False
    )
    print(f"\n  Similarity results saved to: {sim_path}")

    # Save sentiment results
    sent_path = os.path.join(output_dir, "sentiment_results.csv")
    sentiment_df.to_csv(sent_path, index=False)
    print(f"  Sentiment results saved to:  {sent_path}")

    # Save length results
    len_path = os.path.join(output_dir, "length_results.csv")
    length_df.to_csv(len_path, index=False)
    print(f"  Length results saved to:     {len_path}")

    # Save combined summary
    sim_summary = similarity_df.groupby(["Model","Condition"]).agg(
        Mean_Cosine_Distance = ("Cosine_Distance", "mean"),
        SD_Cosine_Distance   = ("Cosine_Distance", "std"),
    ).round(4).reset_index()

    sent_summary = sentiment_df.groupby(["Model","Condition"]).agg(
        Mean_Sentiment_Shift = ("Sentiment_Shift", "mean"),
        SD_Sentiment_Shift   = ("Sentiment_Shift", "std"),
    ).round(4).reset_index()

    len_summary = length_df.groupby(["Model","Condition"]).agg(
        Mean_Word_Difference = ("Word_Difference", "mean"),
        SD_Word_Difference   = ("Word_Difference", "std"),
    ).round(2).reset_index()

    combined = sim_summary.merge(sent_summary, on=["Model","Condition"])
    combined = combined.merge(len_summary,  on=["Model","Condition"])

    comb_path = os.path.join(output_dir, "summary_results.csv")
    combined.to_csv(comb_path, index=False)
    print(f"  Combined summary saved to:   {comb_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 68)
    print("AUTOMATED CONTENT ANALYSIS — GENDER BIAS IN LLM TRAUMA RESPONSES")
    print("=" * 68)

    # Load data
    df = load_data(DATA_PATH)

    # Load sentence embedding model for semantic similarity
    print("\nLoading sentence embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Embedding model loaded.")

    # Run all three analyses
    similarity_df = run_semantic_similarity(df, embed_model)
    print_similarity_summary(similarity_df)

    sentiment_df  = run_sentiment_analysis(df)
    length_df     = run_length_analysis(df)

    # Save results
    print("\n" + "=" * 68)
    print("SAVING RESULTS")
    print("=" * 68)
    save_results(similarity_df, sentiment_df, length_df, OUTPUT_DIR)

    print("\n" + "=" * 68)
    print("ANALYSIS COMPLETE")
    print("=" * 68)
