import re
import string
from collections import Counter

import pandas as pd
from transformers import pipeline


def get_qa_model_name() -> str:
    """Return the QA model name. Provided helper. Do not modify."""
    return "distilbert-base-cased-distilled-squad"


def load_examples(data_path: str) -> pd.DataFrame:
    """
    Load the curated QA CSV.

    Returns a DataFrame with columns: qid, question, context, gold_answer.

    The default `data/tech_news_qa.csv` ships ~1,000 extractive QA tuples
    derived from glnmario/news-qa-summarization (CNN tech / entertainment
    slice). Each gold answer is a literal substring of its context, by
    construction (filtered during curation).

    Provided helper. Do not modify.
    """
    df = pd.read_csv(data_path)
    required = {"qid", "question", "context", "gold_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")
    return df


# -- Task 1: Normalization + EM + F1 (same as drill) -------------------------

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization (see drill / reading)."""
    if not isinstance(s, str):
        return ""
        
    # 1. Lowercase
    s = s.lower()
    
    # 2. Strip standalone articles using word-boundary regex (\b)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    
    # 3. Strip all string.punctuation
    punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
    s = punctuation_pattern.sub(" ", s)
    
    # 4. Collapse whitespace and strip leading/trailing spaces
    s = " ".join(s.split())
    
    return s


def exact_match(pred: str, gold: str) -> int:
    """Return 1 if normalized prediction equals normalized gold."""
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0


def token_f1(pred: str, gold: str) -> float:
    """
    Token-F1 between prediction and gold after normalization.

    Empty handling:
      - both empty -> 1.0
      - one empty -> 0.0
    Returns float in [0.0, 1.0]; never NaN.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    
    # Handle empty cases explicitly to avoid division by zero
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    # Compute multiset overlap using Counter (bag of tokens)
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    
    overlap = sum((pred_counter & gold_counter).values())
    
    # Compute precision, recall, and harmonic mean
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
        
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# -- Task 2: Build the QA pipeline -------------------------------------------

def build_qa_pipeline(model_name: str):
    """Construct a Hugging Face question-answering pipeline."""
    return pipeline("question-answering", model=model_name)


# -- Task 3: Predict one answer ---------------------------------------------

def predict_one(qa, question: str, context: str) -> str:
    """
    Run the QA pipeline on one (question, context) pair.

    Returns the answer STRING only (not the full pipeline output dict).
    """
    result = qa(question=question, context=context)
    return str(result["answer"])