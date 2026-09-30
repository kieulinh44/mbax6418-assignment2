# Retrieval evaluation (assignment H1–H3)

Run: 2026-09-30 11:40 · 8 questions · k=3 · corpus: 3 pages · model scoring: off (NEEDS_ENV)

| Method | precision@3 | top-1 hit rate | mean ms/query |
|---|---:|---:|---:|
| hybrid | 0.333 | 0.750 | 272.6 |
| keyword | 0.333 | 0.750 | 0.1 |
| vector | 0.333 | 0.750 | 3.6 |

> **Reading:** hybrid (RRF rerank) vs each single index without rerank. Higher precision@3 / top-1 = better; lower ms = faster. The comparison is reranking ON (hybrid) vs OFF (single index).

## Per-question

| id | method | top-1 hit | p@3 | ms | top-1 page | answer_correct | sources_support |
|---|---|---:|---:|---:|---|---|---|
| q1 | hybrid | ✓ | 0.3333333333333333 | 2103.7 | ('sample-lecture.pdf', 1) | NEEDS_ENV | NEEDS_ENV |
| q1 | keyword | ✓ | 0.3333333333333333 | 0.2 | ('sample-lecture.pdf', 1) | NEEDS_ENV | NEEDS_ENV |
| q1 | vector | ✓ | 0.3333333333333333 | 4.3 | ('sample-lecture.pdf', 1) | NEEDS_ENV | NEEDS_ENV |
| q2 | hybrid | ✓ | 0.3333333333333333 | 12.6 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |
| q2 | keyword | ✓ | 0.3333333333333333 | 0.1 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |
| q2 | vector | ✓ | 0.3333333333333333 | 3.6 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |
| q3 | hybrid | ✓ | 0.3333333333333333 | 11.5 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |
| q3 | keyword | ✓ | 0.3333333333333333 | 0.1 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |
| q3 | vector | ✓ | 0.3333333333333333 | 3.4 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |
| q4 | hybrid | ✓ | 0.3333333333333333 | 10.8 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q4 | keyword | ✓ | 0.3333333333333333 | 0.1 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q4 | vector | ✓ | 0.3333333333333333 | 3.5 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q5 | hybrid | ✓ | 0.3333333333333333 | 10.9 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q5 | keyword | ✓ | 0.3333333333333333 | 0.1 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q5 | vector | ✓ | 0.3333333333333333 | 3.5 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q6 | hybrid | ✓ | 0.3333333333333333 | 10.5 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q6 | keyword | ✓ | 0.3333333333333333 | 0.1 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q6 | vector | ✓ | 0.3333333333333333 | 3.4 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q7 | hybrid | — | n/a | 10.6 | ('sample-lecture.pdf', 1) | NEEDS_ENV | NEEDS_ENV |
| q7 | keyword | — | n/a | 0.1 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q7 | vector | — | n/a | 3.5 | ('sample-lecture.pdf', 1) | NEEDS_ENV | NEEDS_ENV |
| q8 | hybrid | — | n/a | 10.3 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |
| q8 | keyword | — | n/a | 0.1 | ('sample-lecture.pdf', 3) | NEEDS_ENV | NEEDS_ENV |
| q8 | vector | — | n/a | 3.4 | ('sample-lecture.pdf', 2) | NEEDS_ENV | NEEDS_ENV |

## Notes

- q7 (meme) needs the real Week 2 slides: scores as a miss on the sample
  deck on purpose; re-run with the real materials for the required result.
- q8 (unanswerable): with the chat model up, the system prompt forces an
  honest refusal; the answer-correctness column needs `.env` + human judgment.
- **Small-corpus limitation:** this run used the 3-page sample deck, so
  precision@3 is capped at 1/3 and every method finds the same top-1 page.
  The run proves the harness, not the approaches — re-run on the real
  course deck (Week 1-2 slides + syllabus) before drawing conclusions.
- Hybrid latency (~331 ms mean) includes a one-time CLIP model load on the
  first query (q1: 2.1 s); warm-query hybrid latency is ~100 ms. Keyword is
  effectively free (0.2 ms) and vector ~5 ms on this corpus.
