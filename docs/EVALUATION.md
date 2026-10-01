# Evaluation question set — Course Assistant

A graded set of questions used to check the assistant against the assignment
requirements. Run them with:

```bash
python scripts/eval_questions.py     # needs the app running on http://127.0.0.1:8000
```

Expected behaviors per question are listed below; the runner prints the actual
answer, source files, and whether the answer cited the retrieved evidence.

| # | Category | Question | Expected behavior |
|---|----------|----------|-------------------|
| 1 | Syllabus | What is the grading breakdown for MBAX 6418? | Cites the syllabus page(s) with the percentages (quizzes, final project, final exam, attendance). If a percentage is not extractable, the app must say so instead of guessing. |
| 2 | Syllabus | How is attendance and participation evaluated? | Cites the syllabus attendance criteria (in-person and online criteria). |
| 3 | Slide text (Week 3) | What makes a good few-shot example in prompt engineering? | Grounded answer from the Week 3 deck (few-shot examples slide). |
| 4 | Slide text (Week 5) | What is Retrieval Augmented Generation, and why is it useful? | Cites Week 5 slides (RAG definition + why-use). |
| 5 | Slide text (Week 4) | Why are commits useful when debugging, according to the Week 4 material? | Cites the Week 4 "Commits / Reverting Commits" slides. |
| 6 | Slide text (Week 2) | What is vibe coding? | Cites the Week 2 vibe-coding slides. |
| 7 | **Visual** | What does the flowchart on the "What Is RAG?" slide show? | Retrieves Week 5 slide 10 **by its image**, describes the diagram, and the sources include the actual slide image (doc name + slide number). |
| 8 | **Visual (meme)** | Explain the meme on the "Vibe Coding on Prod" slide (Week 2): which meme format is it and what point does it make? | Retrieves Week 2 slide 33 by its image, identifies the "One does not simply" (Boromir) format, reads the caption ("vibe code a production-grade enterprise app"), and shows the slide image. |
| 9 | **Unanswerable** | What is the professor's favorite programming language? | The materials do not contain this. The app must explicitly acknowledge the information is missing (and not fabricate an answer or citation). |

**Checklist mapped to the assignment:**

- Questions 1–6 cover the **syllabus and slide text**.
- Questions 7–8 require **visual information** (diagram + meme) — the app must
  retrieve the slide *image* through RAG and show it with document name and
  slide number.
- Question 9 verifies the app **acknowledges missing information** instead of
  inventing facts or citations.
- Every answer should return structured `answer` + `sources` fields, and
  sources should tie to real pages/slides.
