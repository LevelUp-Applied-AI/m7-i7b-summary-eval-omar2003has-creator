"""
Module 7 Week B — Integration Task: Summarization & Integrated Evaluation Report.
"""

import json
import os

import pandas as pd
from transformers import pipeline
from rouge_score import rouge_scorer


def get_summarizer_model_name() -> str:
    return os.environ.get("SUMM_MODEL_FOR_CI", "sshleifer/distilbart-cnn-6-6")


def _articles_path() -> str:
    return os.environ.get("ARTICLES_PATH", "data/tech_news_articles.csv")


def _references_path() -> str:
    return os.environ.get("REFERENCES_PATH", "data/tech_news_summaries_reference.csv")


def _output_path() -> str:
    return os.environ.get("OUTPUT_PATH", "summary_predictions.csv")


def build_summarizer(model_name: str):
    return pipeline("summarization", model=model_name)


def summarize_one(summ, text: str, max_length: int = 120, min_length: int = 30) -> str:
    out = summ(text, max_length=max_length, min_length=min_length, do_sample=False, num_beams=4)
    return out[0]["summary_text"]


def compute_rouge(pred: str, ref: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure)
    }


def evaluate_summaries(summ, articles_df: pd.DataFrame, refs_df: pd.DataFrame) -> dict:
    merged_df = pd.merge(articles_df, refs_df, on="article_id")
    predictions = []
    r1_total = r2_total = rl_total = 0.0
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


def main() -> None:
    articles_df = pd.read_csv(_articles_path())
    refs_df = pd.read_csv(_references_path())
    summ = build_summarizer(get_summarizer_model_name())
    result = evaluate_summaries(summ, articles_df, refs_df)
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv(_output_path(), index=False)
    metrics = {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "n": result["n"],
        "model": get_summarizer_model_name(),
    }
    metrics_path = _output_path().replace("predictions", "metrics").replace(".csv", ".json")
    if metrics_path == _output_path():
        metrics_path = "summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"ROUGE-1 = {result['rouge1']:.4f}")
    print(f"ROUGE-2 = {result['rouge2']:.4f}")
    print(f"ROUGE-L = {result['rougeL']:.4f}")


if __name__ == "__main__":
    main()