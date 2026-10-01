# Evaluation results — pending manual review

The canonical question set is [`docs/EVALUATION.md`](EVALUATION.md), also shared by [`scripts/eval_questions.py`](../scripts/eval_questions.py).

Run the comparison after ingesting the syllabus and required Week 2–5 slides:

```bash
python scripts/compare_retrieval.py
```

The script compares `hybrid` (BM25 + text embeddings + CLIP visual fusion) with `text_keyword_only` (BM25 + text embeddings, visual retrieval/fusion disabled). It saves raw results to `docs/evaluation_results.json` and leaves correctness/source-support decisions to human review.

**Evaluation state: Pending evaluation run.** No measured results or final recommendation are included in this template.

## Results table

| Question # | Category | Retrieval approach | Correct? (manual review) | Sources support answer? (manual review) | Validation result | Response time (seconds) | Notes |
|---:|---|---|---|---|---|---:|---|
| 1 | Syllabus | hybrid | — | — | — | — | |
| 1 | Syllabus | text_keyword_only | — | — | — | — | |
| 2 | Syllabus | hybrid | — | — | — | — | |
| 2 | Syllabus | text_keyword_only | — | — | — | — | |
| 3 | Slide text (Week 3) | hybrid | — | — | — | — | |
| 3 | Slide text (Week 3) | text_keyword_only | — | — | — | — | |
| 4 | Slide text (Week 5) | hybrid | — | — | — | — | |
| 4 | Slide text (Week 5) | text_keyword_only | — | — | — | — | |
| 5 | Slide text (Week 4) | hybrid | — | — | — | — | |
| 5 | Slide text (Week 4) | text_keyword_only | — | — | — | — | |
| 6 | Slide text (Week 2) | hybrid | — | — | — | — | |
| 6 | Slide text (Week 2) | text_keyword_only | — | — | — | — | |
| 7 | Visual | hybrid | — | — | — | — | |
| 7 | Visual | text_keyword_only | — | — | — | — | |
| 8 | Visual (meme) | hybrid | — | — | — | — | |
| 8 | Visual (meme) | text_keyword_only | — | — | — | — | |
| 9 | Unanswerable | hybrid | — | — | — | — | Check explicit missing-information acknowledgement. |
| 9 | Unanswerable | text_keyword_only | — | — | — | — | Check explicit missing-information acknowledgement. |

## Summary metrics

Populate after the script runs and a human reviews every row:

- Average response time — hybrid: `—`; text + keyword only: `—`
- Number correct — hybrid: `—`; text + keyword only: `—`
- Number with supported sources — hybrid: `—`; text + keyword only: `—`
- Visual-question performance (Questions 7–8) — hybrid: `—`; text + keyword only: `—`
- Missing-information behavior (Question 9) — hybrid: `—`; text + keyword only: `—`

## Interpretation and conclusion — fill in after review

Likely trade-off to examine, not an evaluation result: hybrid retrieval may be slower but should have an advantage on visual/diagram/meme questions; text-keyword-only retrieval may be faster but can lose visual evidence.

- Observed timing trade-off: `[fill in from evaluation_results.json]`
- Observed visual-question trade-off: `[fill in after manual review of Questions 7–8]`
- Observed source-support trade-off: `[fill in after manual review]`
- Recommended approach: `[fill in only after the complete run and review, or write inconclusive]`
- Rationale: `[cite reviewed question numbers and evidence]`

No correctness, source-support result, or recommendation should be inferred from this template alone.
