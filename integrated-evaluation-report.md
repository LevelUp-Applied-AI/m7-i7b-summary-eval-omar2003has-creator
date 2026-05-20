# Module 7 Integrated Evaluation Report — Fine-Tuning vs. Pre-Trained Inference

> The Module 7 deliverable. Synthesizes Lab 7A (fine-tuning), Integration 7A (domain shift), Lab 7B (QA), and Integration 7B (summarization).

---

## 1. Comparison Table

| Task | Approach | Model | Training cost | Inference cost | Quality metric | Value |
|---|---|---|---|---|---|---|
| Sentiment classification (Lab 7A) | Fine-tuning | DistilBERT | ~30 min CPU + 3K labels | ~50 ms / example | Macro-F1 | **0.6311** |
| Domain transfer (Integration 7A) | Fine-tuned model out-of-domain | (same) | already trained | ~50 ms / example | Domain-shift judgment | Severe performance drop due to technical vocabulary shift |
| Extractive QA (Lab 7B) | Pre-trained inference | distilbert-base-cased-distilled-squad | 0 | ~50 ms / example | EM / token-F1 | **EM: 0.3440 / Token-F1: 0.4705** |
| Summarization (Integration 7B) | Pre-trained inference | distilbart-cnn-6-6 | 0 | ~3 sec / example | ROUGE-1 / 2 / L F1 | **R1: 0.3683 / R2: 0.1572 / RL: 0.2670** |

## 2. Findings

- **Fine-Tuning Trade-offs:** Fine-tuning on in-domain data (Lab 7A) achieved a baseline Macro-F1 of **0.6311** on app store reviews, but completely failed to generalize under domain shift (Integration 7A) when tested on tech news, resulting in misclassifications due to the extreme shift in background vocabulary.
- **Pre-trained Extraction Limits:** Pre-trained extractive QA (Lab 7B) provides accurate results on structured text, but its **EM of 0.3440** and **Token-F1 of 0.4705** show its vulnerability when input contexts stray from standard SQuAD phrasing or introduce noisy formatting.
- **Abstractive Bottlenecks:** Pre-trained abstractive summarization (Integration 7B) maintains high sentence fluency, but suffers from significantly high CPU latency (~3 seconds per article) and struggles to retain specific multi-word technical terminology, as shown by the lower ROUGE-2 score (**0.1572**).

## 3. Faithfulness Check

### Example A — high ROUGE
> **Article excerpt:** `Javier Bardem confirmed that he would be playing the bad guy in "Bond 23" "I am very excited. So to play that is going to be fun," he said . "They chose me to play this man, but I cannot give you many details"`
> **Predicted summary:** `Javier Bardem confirmed that he would be playing the bad guy in "Bond 23" "I am very excited. So to play that is going to be fun," he said .`
> **ROUGE-1:** `0.5833`; **ROUGE-2:** `0.4000`; **ROUGE-L:** `0.5000`
> **Faithful?** Yes. The summary is 100% faithful to the text because the model successfully preserved the primary core identity and quotes without introducing external noise or modification.

### Example B — mid ROUGE
> **Article excerpt:** `Actor Derek Mears takes on role of Jason in new "Friday the 13th" Actor heard from producers that he was a popular casting choice . Of iconic role, Mears says he tried to "definitely make it my own"Actor Derek Mears stars as Jason Voorhees in new "Friday the 13th" movie . Mears: "I've got a lot of daddy issues. No, I sound like a basket case. It's funny with acting -- we all wear masks in our normal life," Mears says .`
> **Predicted summary:** `Actor Derek Mears stars as Jason Voorhees in new "Friday the 13th" movie .`
> **ROUGE-1:** `0.3456`; **ROUGE-2:** `0.1772`; **ROUGE-L:** `0.3209`
> **Faithful?** Yes. The summary is completely accurate and faithful, but ROUGE penalized it heavily with a mid-tier score simply because the model extracted a highly concise single-sentence summary, which drastically reduced the token overlap against a longer reference summary.

### Example C — low ROUGE
> **Article excerpt:** `Tennis stars of past and present have turned their talents to land roles in films . American ace John McEnroe has appeared in the credits for no less than six features . 1957 Wimbledon champion starred alongside screen icon John Wayne in a western . Indian star Vijay Amritraj was talent spotted for a supporting part in Bond movie Octopussy . Rafael Nadal appeared with Colombian rock star Shakira in a new music video . The scenes of the two stars used to accompany the global release of the "Gypsy" single were filmed in Barcelona, Spain . Nadal is in no danger of winning an Oscar for his acting skills .`
> **Predicted summary:** `Tennis stars of past and present have turned their talents to land roles in films .`
> **ROUGE-1:** `0.2500`; **ROUGE-2:** `0.0196`; **ROUGE-L:** `0.1346`
> **Faithful?** Yes. While the summary is factual and faithful, it scored a very low ROUGE-2 of **0.0196** because the model only caught the introductory background sentence and completely missed all the specific multi-word technical names and entities (like John McEnroe, Rafael Nadal, Shakira) that were highlighted in the human reference summary.

## 4. Production Decision Matrix

| Scenario | Recommendation | Justification |
|---|---|---|
| Real-time app store review triage dashboard for a product team | **Fine-tuning** | Fine-tuning is required here to meet the strict low-latency budget (~50 ms/example) and maintain the specialized vocabulary classification baseline proven by our 0.6311 Macro-F1 score. |
| Daily tech / entertainment news summary digest for an internal newsroom | **Pre-trained inference** | Pre-trained inference is ideal because an offline daily digest can easily tolerate the ~3 seconds/example CPU latency to leverage a baseline ROUGE-1 of 0.3683 without data-labeling costs. |
| Domain-expert QA on legal contracts | **Pre-trained inference** | Pre-trained inference with an extractive framework must be used to guarantee safety, anchoring answers strictly to verbatim text spans and avoiding abstractive hallucinations. |

## 5. What You Would Do Differently

If a fully labeled summarization dataset for this domain becomes available, my primary engineering investment would be to transition from pre-trained zero-shot inference to downstream task fine-tuning on a sequence-to-sequence architecture (such as BART or T5). Fine-tuning on domain-specific tech articles would directly optimize the decoder for technical entity retention, driving up the baseline ROUGE-2 and ROUGE-L scores. Furthermore, I would implement a classification-based calibration model on top of the decoder's token probabilities to automatically filter and flag low-confidence outputs, creating a production gate against any potential abstractive hallucinations.

## 6. Limits of the Evaluation

These measured evaluation metrics leave critical operational blind spots unaddressed. First, the ROUGE metrics do not account for factual faithfulness; a model can score highly on token overlap while generating clean, coherent lies that completely invert the article's true meaning. Second, our extractive QA metrics (**EM of 0.3440**) do not capture system calibration, meaning we cannot detect if a model is outputting errors with absolute confidence. Finally, our latency tracking is limited to single-request operations on CPU; it fails to simulate concurrent traffic loads, queue delays, or memory saturation thresholds under real production pipeline stress.