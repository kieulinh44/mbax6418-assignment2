# Evaluation results

The canonical question set is [`docs/EVALUATION.md`](EVALUATION.md), also shared by [`scripts/eval_questions.py`](../scripts/eval_questions.py).

Run the comparison after ingesting the syllabus and required Week 2–5 slides:

```bash
python scripts/compare_retrieval.py
```

The script compares `hybrid` (BM25 + text embeddings + CLIP visual fusion) with `text_keyword_only` (BM25 + text embeddings, visual retrieval/fusion disabled). It saves raw results to `docs/evaluation_results.json` and leaves correctness/source-support decisions to human review.

**Evaluation run:** October 1, 2026. The syllabus and Weeks 2–5 slide decks were
indexed locally before the comparison. Each canonical question was run once in
each retrieval mode. Correctness and source support below were manually
reviewed against the course materials.

## Results table

| Question # | Category | Retrieval approach | Correct? (manual review) | Sources support answer? (manual review) | Validation result | Response time (seconds) | Notes |
|---:|---|---|---|---|---|---:|---|
| 1 | Syllabus | hybrid | Yes | Yes | False | 13.497 | Manual review confirmed the complete grading breakdown; automatic validation did not match every citation. |
| 1 | Syllabus | text_keyword_only | Yes | Yes | False | 10.939 | Manual review confirmed the complete grading breakdown; automatic validation did not match every citation. |
| 2 | Syllabus | hybrid | Yes | Yes | True | 7.455 | Attendance/participation weight and criteria were supported. |
| 2 | Syllabus | text_keyword_only | Yes | Yes | True | 6.704 | Attendance/participation weight and criteria were supported. |
| 3 | Slide text (Week 3) | hybrid | Yes | Yes | True | 6.422 | Few-shot explanation was supported by Week 3 examples. |
| 3 | Slide text (Week 3) | text_keyword_only | Yes | Yes | True | 12.927 | Few-shot explanation was supported by Week 3 examples. |
| 4 | Slide text (Week 5) | hybrid | Yes | Yes | True | 5.524 | RAG definition and purpose were supported by Week 5 slides. |
| 4 | Slide text (Week 5) | text_keyword_only | Yes | Yes | True | 5.686 | RAG definition and purpose were supported by Week 5 slides. |
| 5 | Slide text (Week 4) | hybrid | Yes | Yes | False | 11.832 | Commits explanation was manually supported; automated citation matching was incomplete. |
| 5 | Slide text (Week 4) | text_keyword_only | Yes | Yes | False | 9.367 | Commits explanation was manually supported; automated citation matching was incomplete. |
| 6 | Slide text (Week 2) | hybrid | Yes | Yes | False | 10.372 | Vibe-coding explanation was manually supported; automated citation matching was incomplete. |
| 6 | Slide text (Week 2) | text_keyword_only | Yes | Yes | False | 9.218 | Vibe-coding explanation was manually supported; automated citation matching was incomplete. |
| 7 | Visual | hybrid | Yes | Yes | True | 6.157 | Correctly explained the Simplified RAG flowchart. |
| 7 | Visual | text_keyword_only | Yes | Yes | False | 5.053 | Correctly explained the flowchart; automatic citation matching was incomplete. |
| 8 | Visual (meme) | hybrid | Yes | Yes | False | 6.121 | Correctly identified and explained the Boromir meme; automatic citation matching was incomplete. |
| 8 | Visual (meme) | text_keyword_only | Yes | Yes | True | 22.162 | Correctly identified and explained the Boromir meme. |
| 9 | Unanswerable | hybrid | Yes | Yes | True | 3.944 | Correctly acknowledged that the materials do not provide the information. |
| 9 | Unanswerable | text_keyword_only | Yes | Yes | True | 5.414 | Correctly acknowledged that the materials do not provide the information. |

## Summary metrics

The timing results come from the local comparison run; correctness and source
support were manually reviewed.

- Average response time — hybrid: `7.925 seconds`; text + keyword only: `9.719 seconds`
- Number correct — hybrid: `9 / 9`; text + keyword only: `9 / 9`
- Number with supported sources — hybrid: `9 / 9`; text + keyword only: `9 / 9`
- Visual-question performance (Questions 7–8) — both approaches were correct; hybrid averaged `6.139 seconds` and text + keyword only averaged `13.608 seconds`
- Missing-information behavior (Question 9) — both approaches explicitly acknowledged that the information was not in the materials

## Interpretation and conclusion

Both approaches answered all nine reviewed questions correctly and used
manually supported sources. The hybrid approach was faster overall (7.925
seconds versus 9.719 seconds) and showed its clearest benefit on the visual
meme question: 6.121 seconds versus 22.162 seconds. The two approaches were
similarly successful on ordinary syllabus and slide-text questions.

We would keep **hybrid retrieval**. It preserves the ability to use visual
slide evidence for diagrams and memes, while this evaluation run also showed a
lower average response time. Automated citation validation was false on some
otherwise supported answers, so the source-support judgments above reflect
manual review rather than the validation flag alone.
