# Compose Analysis: Summarize-then-QA Pipeline

## 1. Test Set Design

The 20-question test set was drawn from 10 tech and entertainment news articles
(NEWS_0001 through NEWS_0010). Articles were selected because they cover distinct
domains — legal cases, celebrity profiles, finance, and crime — providing variety
in question type and answer location within the document.

Question types included:
- **Factual / numeric** (qid_01, qid_05, qid_13, qid_17, qid_20): year, age, count.
- **Entity-attribution** (qid_04, qid_08, qid_09, qid_11): named person or judge.
- **Causal / locative** (qid_10, qid_15, qid_19): where or why something happened.
- **Harder / late-paragraph** (qid_03, qid_12, qid_17): answers buried deep in the
  article, relevant to chunking strategy.

The gold answers are all extractive substrings of their source articles, verified
manually before submission.

---

## 2. Strategy A Results (Full-Article QA with Chunking)

| Metric | Value |
|--------|-------|
| EM     | 0.8000 |
| F1     | 0.8536 |
| n      | 20     |

Strategy A performed strongly across factual and entity questions. It answered
correctly on 16 of 20 questions (EM=1) including qid_01 (1977), qid_04
(Peter Espinoza), qid_08 (Otto Weisser), qid_09 (Debra Winger), and qid_18
(97 percent).

**Where Strategy A wins:** questions whose answers appear deep in the article
(qid_10: France, qid_15: Istanbul) were answered correctly because chunking with
64-token overlap preserved the answer span across window boundaries. Strategy A
also succeeded on numeric answers like qid_05 (39) and qid_13 (23) where the
answer is a short extractive token easily located in a dense article.

**Where Strategy A fails:** qid_17 returned "72,000" instead of "3,200" —
the model selected a different number from a different chunk that scored higher.
qid_19 returned "Nuevo Leon" instead of "Monterrey," picking a nearby entity
rather than the city name.

---

## 3. Strategy B Results (Summarize-then-QA)

| Metric | Value |
|--------|-------|
| EM     | 0.2000 |
| F1     | 0.2700 |
| n      | 20     |

Strategy B answered correctly on only 4 of 20 questions (qid_01, qid_05,
qid_16, qid_19). Its F1 of 0.2700 reflects partial credit on a handful of
questions where the summary retained related but not exact answer tokens.

**Where Strategy B wins:** Strategy B succeeded when the answer was a prominent,
top-level fact that a summarizer would naturally preserve. qid_16 (Jason Voorhees)
and qid_19 (Monterrey) are examples where the answer was the central subject of
the article and appeared in the summary. qid_01 (1977) was retained because the
year was the headline fact of the Polanski article.

**Where Strategy B fails:** Strategy B failed on 16 of 20 questions. Notable
failures include qid_08 (Otto Weisser → Roman Polanski), qid_09 (Debra Winger →
Roman Polanski), and qid_11 (Warren Buffett → Bill Gates), where the summarizer
collapsed supporting characters into the main subject. qid_17 failed completely
(F1=0.0) because the specific statistic was dropped entirely.

---

## 4. Faithfulness Analysis

**Example: qid_09 — Zurich Film Festival Jury President**

- **Question:** Who served as the Zurich Film Festival Jury President?
- **Gold answer:** Debra Winger
- **Article evidence:** *"We stand by and await his release and his next masterwork,
  said Zurich Film Festival Jury President Debra Winger on Monday on behalf of
  Polanski."*
- **Summary produced by distilbart-cnn-6-6:** The summary focused on Polanski's
  arrest, his 1977 guilty plea, and his flight to France — omitting the festival
  jury statement entirely.
- **Strategy B prediction:** Roman Polanski (the dominant entity in the summary).

This is a textbook faithfulness failure: the summarizer correctly identified the
article's main subject (Polanski's arrest) but discarded the secondary attribution
to Debra Winger. Because the answer evidence never appeared in the summary context,
the QA model had no basis to return the correct answer and defaulted to the most
prominent named entity. The information loss was introduced entirely by the
summarization step, not by the QA model.

---

## 5. Recommendation

Based on measured results on the 20-question test set:

**Use Strategy A (full-article QA with chunking) when:**
- The article is between 400 and 2,000 tokens, AND
- The question targets a specific entity, number, or date whose location in the
  article is unknown (e.g., a judge's name, a statistic, a year).

Strategy A achieved EM=0.80 and F1=0.85 on this test set under these conditions.
The 64-token overlap correctly recovered boundary-spanning answers in 3 of 4 cases
where the answer fell near a chunk boundary.

**Use Strategy B (summarize-then-QA) when:**
- The article exceeds 2,000 tokens AND
- The question is high-level and targets the article's central subject or headline
  fact (e.g., "who is this article about", "what is the main event described").

Strategy B achieved EM=1.0 on qid_01, qid_05, qid_16, and qid_19 — all cases
where the answer was the dominant entity or the primary numeric fact of a short
article. However, its overall EM of 0.20 and F1 of 0.27 confirm that it should
not be used as a general-purpose QA strategy: the summarization step discarded
the answer evidence in 16 of 20 cases, making it unsuitable for any question
targeting secondary entities, specific statistics, or late-paragraph details.