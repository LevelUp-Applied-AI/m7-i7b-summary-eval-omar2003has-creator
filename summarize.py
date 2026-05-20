"""
Module 7 Week B — Integration Task: Summarization & Integrated Evaluation Report.

Implement the functions below. See the integration guide for full task descriptions.

The integrated evaluation report (`integrated-evaluation-report.md`) is the M7
deliverable. Write it by hand based on the metrics produced by main().
"""

import json
import os

import pandas as pd
from transformers import pipeline
from rouge_score import rouge_scorer


# -- Helpers (provided — do NOT modify) --------------------------------------

def get_summarizer_model_name() -> str:
    """Return env override (CI smoke) or the default summarization model."""
    return os.environ.get("SUMM_MODEL_FOR_CI", "sshleifer/distilbart-cnn-6-6")


def _articles_path() -> str:
    return os.environ.get("ARTICLES_PATH", "data/tech_news_articles.csv")


def _references_path() -> str:
    return os.environ.get("REFERENCES_PATH", "data/tech_news_summaries_reference.csv")


def _output_path() -> str:
    return os.environ.get("OUTPUT_PATH", "summary_predictions.csv")


# -- Task 1: Pipeline + single-document summarization ------------------------

def build_summarizer(model_name: str):
    """Construct a Hugging Face summarization pipeline."""
    return pipeline("summarization", model=model_name)


def summarize_one(summ, text: str, max_length: int = 120, min_length: int = 30) -> str:
    """
    Summarize one document with deterministic beam search.

    Use do_sample=False, num_beams=4. Return the summary STRING from
    [0]["summary_text"].
    """
    out = summ(
        text, 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False, 
        num_beams=4
    )
    return out[0]["summary_text"]


# -- Task 2: ROUGE -----------------------------------------------------------

def compute_rouge(pred: str, ref: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1.

    """
    #  use_stemmer=True  It returns words to their linguistic root (for example, the words: running, runs, ran completely are unified and interpreted as run).
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure)
    }


# -- Task 3: Evaluate over the corpus ----------------------------------------

def evaluate_summaries(summ, articles_df: pd.DataFrame, refs_df: pd.DataFrame) -> dict:
    """
    Summarize each article and score against its reference.


    """
    merged_df = pd.merge(articles_df, refs_df, on="article_id")
    
    predictions = []
    r1_total = 0.0
    r2_total = 0.0
    rl_total = 0.0
    n = len(merged_df)
    
    for _, row in merged_df.iterrows():
        article_id = str(row["article_id"])
        text = str(row["text"])
        ref_summary = str(row["reference_summary"])
        
        pred_summary = summarize_one(summ, text)
        scores = compute_rouge(pred_summary, ref_summary)
        
        r1_total += scores["rouge1"]
        r2_total += scores["rouge2"]
        rl_total += scores["rougeL"]
        
        predictions.append({
            "article_id": article_id,
            "reference_summary": ref_summary,
            "predicted_summary": pred_summary,
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"]
        })
        
    return {
        "rouge1": float(r1_total / n) if n > 0 else 0.0,
        "rouge2": float(r2_total / n) if n > 0 else 0.0,
        "rougeL": float(rl_total / n) if n > 0 else 0.0,
        "n": int(n),
        "predictions": predictions
    }


# -- Task 4: Orchestrate -----------------------------------------------------

def main() -> None:
    """Load data, build pipeline, evaluate, write artifacts."""
    articles_df = pd.read_csv(_articles_path())
    refs_df = pd.read_csv(_references_path())

    summ = build_summarizer(get_summarizer_model_name())
    result = evaluate_summaries(summ, articles_df, refs_df)

    # Write predictions CSV
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv(_output_path(), index=False)

    # Write metrics JSON
    metrics = {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "n": result["n"],
        "model": get_summarizer_model_name(),
    }
    metrics_path = _output_path().replace("predictions", "metrics").replace(".csv", ".json")
    if metrics_path == _output_path():  # safety: ensure rename happened
        metrics_path = "summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ROUGE-1 = {result['rouge1']:.4f}")
    print(f"ROUGE-2 = {result['rouge2']:.4f}")
    print(f"ROUGE-L = {result['rougeL']:.4f}")
    print(f"n = {result['n']}")


if __name__ == "__main__":
    main()