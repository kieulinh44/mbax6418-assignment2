"""Deterministic hybrid retrieval tests; no model downloads or LLM calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app import hybrid


class SlideRetrievalTests(unittest.TestCase):
    def setUp(self):
        # Bypass persistent indexes and model loading, not retrieval logic.
        self.corpus = hybrid.Corpus.__new__(hybrid.Corpus)
        self.corpus.pages = [
            {"doc": doc, "file": doc + ".pptx", "page": page,
             "kind": "pptx", "text": text, "image": f"{doc}-{page}.png"}
            for doc, page, text in [
                ("lecture", 1, "overview"),
                ("lecture", 12, "target economics"),
                ("other", 12, "other economics"),
                ("lecture", 3, "third economics"),
            ]
        ]
        self.corpus.chunks = [
            {**page, "cid": f"{page['doc']}__p{page['page']}__0",
             "section": "economics" if i else "overview",
             "page_index": i}
            for i, page in enumerate(self.corpus.pages)
        ]
        self.corpus.retriever = Mock()
        self.corpus.retriever.retrieve.return_value = SimpleNamespace(
            documents=np.array([[0, 2, 3, 1]]),
            scores=np.array([[10.0, 8.0, 5.0, 1.0]]),
        )
        self.corpus.text_enc = Mock()
        self.corpus.text_enc.encode.return_value = np.array([0.1, 0.2])
        self.corpus.text_coll = Mock()
        self.corpus.text_coll.query.return_value = {
            "ids": [[self.corpus.chunks[i]["cid"] for i in [0, 2, 3, 1]]],
            "distances": [[0.01, 0.1, 0.2, 0.9]],
        }
        self.corpus.visual_coll = Mock()
        self.corpus.visual_coll.count.return_value = 4
        self.corpus.visual_coll.query.return_value = {
            "ids": [[hybrid._visual_id(self.corpus.pages[i]) for i in [0, 2, 3, 1]]],
            "distances": [[0.01, 0.1, 0.2, 0.9]],
        }
        self.clip = Mock()
        self.clip.encode.return_value = np.array([0.3, 0.4])
        self.clip_patch = patch.object(hybrid, "_clip_model", return_value=self.clip)
        self.clip_patch.start()
        self.addCleanup(self.clip_patch.stop)

    def retrieve(self, query, **kwargs):
        return self.corpus._retrieve_unlocked(query, top_k=10, **kwargs)

    def requirement_2_fixture(self):
        """Use real requirement labels and the real keyword retriever."""
        import bm25s

        doc = "mbax-6418-week-2-llm-fundamentals-v2"
        filename = "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx"
        rows = [
            (7, "Benchmark Evaluation Intelligence Cost per Task"),
            (33, 'Vibe Coding on "Prod"'),
            (27, "Vibe Coding Overview"),
        ]
        self.corpus.pages = [
            {"doc": doc, "file": filename, "page": page, "kind": "pptx",
             "text": text, "image": f"week2-{page}.png"}
            for page, text in rows
        ]
        self.corpus.chunks = [
            {**page, "cid": f"{doc}__p{page['page']}__0", "section": None,
             "page_index": i}
            for i, page in enumerate(self.corpus.pages)
        ]
        tokens = bm25s.tokenize([chunk["text"] for chunk in self.corpus.chunks],
                                stopwords="en")
        self.corpus.retriever = bm25s.BM25()
        self.corpus.retriever.index(tokens)
        self.corpus.text_coll.query.return_value = {
            "ids": [[]], "distances": [[]],
        }
        self.corpus.visual_coll.count.return_value = 0
        return doc, filename

    def test_explicit_slide_survives_each_indexes_global_cutoff(self):
        # The target ranks 126th: outside BM25's 100, CLIP's 20, and
        # text retrieval's one-result limit for a single candidate chunk.
        self.corpus.pages = [
            {"doc": "lecture", "file": "lecture.pptx", "page": page,
             "kind": "pptx", "text": f"economics {page}",
             "image": f"lecture-{page}.png"}
            for page in range(1, 127)
        ]
        self.corpus.chunks = [
            {**page, "cid": f"lecture__p{page['page']}__0",
             "section": "economics", "page_index": i}
            for i, page in enumerate(self.corpus.pages)
        ]

        def matches(row, where):
            if not where:
                return True
            if "$and" in where:
                return all(matches(row, clause) for clause in where["$and"])
            return all(row.get(key) == value for key, value in where.items())

        def vector_query(rows, id_fn, **kwargs):
            # Chroma applies metadata filtering BEFORE its result limit.
            selected = [row for row in rows if matches(row, kwargs.get("where"))]
            selected = selected[:kwargs["n_results"]]
            return {"ids": [[id_fn(row) for row in selected]],
                    "distances": [[0.5] * len(selected)]}

        def keyword_query(tokens, k):
            return SimpleNamespace(documents=np.array([list(range(k))]),
                                   scores=np.array([[1.0] * k]))

        for stage in ["keyword", "text", "visual"]:
            for doc in ["lecture", None]:
                with self.subTest(stage=stage, doc=doc):
                    self.corpus.retriever.retrieve.side_effect = (
                        keyword_query if stage == "keyword" else
                        lambda *args, **kwargs: SimpleNamespace(
                            documents=[[]], scores=[[]]))
                    self.corpus.text_coll.query.side_effect = (
                        lambda **kwargs: vector_query(
                            self.corpus.chunks, lambda row: row["cid"], **kwargs)
                        if stage == "text" else {"ids": [[]], "distances": [[]]})
                    self.corpus.visual_coll.count.return_value = len(self.corpus.pages)
                    self.corpus.visual_coll.query.side_effect = (
                        lambda **kwargs: vector_query(
                            self.corpus.pages, hybrid._visual_id, **kwargs)
                        if stage == "visual" else {"ids": [[]], "distances": [[]]})
                    hits = self.retrieve("Describe the image on slide 126", doc=doc,
                                         topic="economics")
                    self.assertEqual(
                        [(hit["page"]["doc"], hit["page"]["page"]) for hit in hits],
                        [("lecture", 126)])
                    if stage != "keyword":
                        collection = (self.corpus.text_coll if stage == "text"
                                      else self.corpus.visual_coll)
                        expected = ({"$and": [{"doc": doc}, {"page": 126}]}
                                    if doc else {"page": 126})
                        self.assertEqual(collection.query.call_args.kwargs["where"],
                                         expected)

    def test_absent_target_returns_empty_without_searching_other_pages(self):
        self.assertEqual(self.retrieve("Explain slide 99", doc="lecture"), [])
        self.corpus.retriever.retrieve.assert_not_called()
        self.corpus.text_coll.query.assert_not_called()
        self.corpus.visual_coll.query.assert_not_called()

    def test_target_intersects_topic_filter(self):
        self.assertEqual(self.retrieve("Explain slide 12", doc="lecture",
                                       topic="overview"), [])
        hits = self.retrieve("Explain slide 12", doc="lecture", topic="economics")
        self.assertEqual([h["page"]["page"] for h in hits], [12])

    def test_without_document_filter_all_matching_documents_remain_candidates(self):
        hits = self.retrieve("Explain slide 12")
        self.assertEqual({(h["page"]["doc"], h["page"]["page"]) for h in hits},
                         {("lecture", 12), ("other", 12)})

    def test_image_query_cannot_reintroduce_wrong_page_or_document(self):
        hits = self.retrieve("Describe the image on slide #12", doc="other")
        self.assertEqual([(h["page"]["doc"], h["page"]["page"]) for h in hits],
                         [("other", 12)])

    def test_ordinary_queries_keep_semantic_ranking(self):
        hits = self.retrieve("Explain economics", doc="lecture")
        self.assertEqual([h["page"]["page"] for h in hits], [1, 3, 12])

    def test_visual_only_target_does_not_bypass_topic_chunk_filter(self):
        excluded = {**self.corpus.chunks[1], "cid": "lecture__p12__excluded",
                    "text": "unrelated notes", "section": "unrelated"}
        self.corpus.chunks.insert(1, excluded)
        self.corpus.retriever.retrieve.return_value = SimpleNamespace(
            documents=np.array([[0, 1]]), scores=np.array([[10.0, 8.0]]))
        self.corpus.text_coll.query.return_value = {
            "ids": [[excluded["cid"]]], "distances": [[0.01]]}
        hits = self.retrieve("Describe the image on slide 12", doc="lecture",
                             topic="economics")
        self.assertEqual([h["chunk"] for h in hits], ["target economics"])

    def test_ambiguous_references_leave_retrieval_unfiltered(self):
        # Multi-reference, lists and ranges are deliberately NOT targeted.
        for query in ["Compare slide 1 and slide 12", "Compare slide 1 and page 3",
                      "Explain slide 1-12", "Explain slide 1 to 12",
                      "Explain slide 1 and 12", "Explain slide 1, 12",
                      "Explain slide 1–12", "Explain slide 1.5"]:
            with self.subTest(query=query):
                hits = self.retrieve(query, doc="lecture")
                self.assertEqual({h["page"]["page"] for h in hits}, {1, 3, 12})

    def test_supported_singular_reference_forms(self):
        for query, number in [("Explain slide #12", 12),
                              ("Explain slide number 12", 12),
                              ("Explain PAGE 3", 3)]:
            with self.subTest(query=query):
                hits = self.retrieve(query, doc="lecture")
                self.assertEqual([h["page"]["page"] for h in hits], [number])

    def test_explicit_slide_overrides_other_pages_higher_scores(self):
        hits = self.retrieve("Explain slide 12", doc="lecture")
        self.assertEqual([(h["page"]["doc"], h["page"]["page"]) for h in hits],
                         [("lecture", 12)])
        # Real retrieval still crosses every index/encoder boundary.
        self.corpus.retriever.retrieve.assert_called_once()
        self.corpus.text_enc.encode.assert_called_once_with("Explain slide 12")
        self.corpus.text_coll.query.assert_called_once()
        self.clip.encode.assert_called_once_with("Explain slide 12")
        self.corpus.visual_coll.query.assert_called_once()

    def test_requirement_2_meme_query_retrieves_week2_slide_33_first(self):
        doc, filename = self.requirement_2_fixture()
        hits = self.retrieve('Find the meme about Vibe Coding on "Prod".', doc=doc)
        self.assertEqual((hits[0]["page"]["file"], hits[0]["page"]["page"]),
                         (filename, 33))
        self.assertEqual(hits[0]["page"]["image"], "week2-33.png")

    def test_requirement_2_chart_query_retrieves_week2_slide_7_first(self):
        doc, filename = self.requirement_2_fixture()
        hits = self.retrieve(
            "Find the Benchmark Evaluation Intelligence and Cost per Task charts.",
            doc=doc,
        )
        self.assertEqual((hits[0]["page"]["file"], hits[0]["page"]["page"]),
                         (filename, 7))
        self.assertEqual(hits[0]["page"]["image"], "week2-7.png")


if __name__ == "__main__":
    unittest.main()
