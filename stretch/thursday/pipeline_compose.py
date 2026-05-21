"""
Module 7 Week B — Thursday Stretch (Honors): Summarize-then-QA.
"""

import importlib.util
import json
import os
import sys

import pandas as pd

# ── Import summarize.py from project root ──────────────────────────────────
_root = os.getcwd()
_summ_path = os.path.join(_root, "summarize.py")
_spec = importlib.util.spec_from_file_location("summarize", _summ_path)
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

# ── Import qa_utils ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(_root, "stretch", "thursday"))
try:
    import qa_utils
except ImportError as e:
    raise ImportError(
        "qa_utils.py not found. Copy build_qa_pipeline, predict_one, exact_match, "
        "token_f1, normalize_answer, and get_qa_model_name from your Lab 7B lab.py "
        "into stretch/thursday/qa_utils.py."
    ) from e


def qa_full_article(qa, question: str, article: str, max_chunk: int = 384) -> str:
    words = article.split()

    if len(words) <= max_chunk:
        return qa_utils.predict_one(qa, question, article)

    stride = max_chunk - 64
    best_answer = ""
    best_score = -1.0

    start = 0
    while start < len(words):
        end = min(start + max_chunk, len(words))
        chunk = " ".join(words[start:end])

        result = qa(question=question, context=chunk)
        score = result.get("score", 0.0)
        answer = result.get("answer", "")

        if score > best_score:
            best_score = score
            best_answer = answer

        if end == len(words):
            break
        start += stride

    return best_answer


def qa_via_summary(qa, summ, question: str, article: str, max_summary_length: int = 120) -> str:
    summary_text = summarize.summarize_one(
        summ,
        article,
        max_length=max_summary_length,
        min_length=30,
    )
    answer = qa_utils.predict_one(qa, question, summary_text)
    return answer


def evaluate_strategies(qa, summ, test_set: pd.DataFrame, articles_df: pd.DataFrame) -> dict:
    """
    Run both strategies on every row of the test set; compute per-strategy EM/F1.

    Returns:
        {
          "strategy_a": {"em": float, "f1": float, "n": int},
          "strategy_b": {"em": float, "f1": float, "n": int},
          "predictions": [
            {qid, question, strategy_a_pred, strategy_b_pred, gold_answer,
             strategy_a_em, strategy_a_f1, strategy_b_em, strategy_b_f1},
            ...
          ],
        }
    """
    predictions = []
    a_em_total = a_f1_total = b_em_total = b_f1_total = 0.0
    n = len(test_set)

    for _, row in test_set.iterrows():
        qid = str(row["qid"])
        article_id = str(row["article_id"])
        question = str(row["question"])
        gold = str(row["gold_answer"])

        article_row = articles_df[articles_df["article_id"].astype(str) == article_id]
        article_text = str(article_row.iloc[0]["text"]) if not article_row.empty else ""

        pred_a = qa_full_article(qa, question, article_text)
        pred_b = qa_via_summary(qa, summ, question, article_text)

        a_em = qa_utils.exact_match(pred_a, gold)
        a_f1 = qa_utils.token_f1(pred_a, gold)
        b_em = qa_utils.exact_match(pred_b, gold)
        b_f1 = qa_utils.token_f1(pred_b, gold)

        a_em_total += a_em
        a_f1_total += a_f1
        b_em_total += b_em
        b_f1_total += b_f1

        predictions.append({
            "qid": qid,
            "question": question,
            "strategy_a_pred": pred_a,
            "strategy_b_pred": pred_b,
            "gold_answer": gold,
            "strategy_a_em": a_em,
            "strategy_a_f1": round(a_f1, 4),
            "strategy_b_em": b_em,
            "strategy_b_f1": round(b_f1, 4),
        })

    return {
        "strategy_a": {
            "em": round(a_em_total / n, 4) if n > 0 else 0.0,
            "f1": round(a_f1_total / n, 4) if n > 0 else 0.0,
            "n": n,
        },
        "strategy_b": {
            "em": round(b_em_total / n, 4) if n > 0 else 0.0,
            "f1": round(b_f1_total / n, 4) if n > 0 else 0.0,
            "n": n,
        },
        "predictions": predictions,
    }

def main() -> None:
    test_set = pd.read_csv("stretch/thursday/qa_test_set.csv")
    articles_df = pd.read_csv("data/tech_news_articles.csv")

    qa = qa_utils.build_qa_pipeline(qa_utils.get_qa_model_name())
    summ = summarize.build_summarizer(summarize.get_summarizer_model_name())

    result = evaluate_strategies(qa, summ, test_set, articles_df)

    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("stretch/thursday/compose_predictions.csv", index=False)

    metrics = {
        "strategy_a": result["strategy_a"],
        "strategy_b": result["strategy_b"],
        "qa_model": qa_utils.get_qa_model_name(),
        "summarizer_model": summarize.get_summarizer_model_name(),
    }
    with open("stretch/thursday/compose_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Strategy A (full-article QA)   — EM={result['strategy_a']['em']:.4f}, F1={result['strategy_a']['f1']:.4f}")
    print(f"Strategy B (summarize-then-QA) — EM={result['strategy_b']['em']:.4f}, F1={result['strategy_b']['f1']:.4f}")


if __name__ == "__main__":
    main()