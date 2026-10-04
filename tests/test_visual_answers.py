"""Image-grounded answer contracts; only model/retrieval boundaries are mocked."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app import main


class VisualAnswersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = Path(self.temp.name)
        (self.index / "pages").mkdir()
        self.addCleanup(patch.stopall)
        patch.object(main.config, "DATA_INDEX", str(self.index)).start()
        self.corpus = patch.object(main.hybrid, "get_corpus").start().return_value
        self.chat = patch.object(main.llm, "chat", return_value="Grounded answer").start()
        self.ground = patch.object(main.llm, "ground", return_value="Grounded answer").start()
        self.vision = patch.object(main.llm, "vision", side_effect=self.describe).start()

    @staticmethod
    def describe(prompt, paths):
        return "Observed " + Path(paths[0]).name

    def hit(self, number, image=True, kind="pptx", file="lecture.pptx"):
        path = f"pages/slide-{number}.png" if image else None
        if image:
            (self.index / path).write_bytes(b"mock image boundary fixture")
        return {"page": {"doc": "lecture", "file": file, "page": number,
                         "kind": kind, "image": path, "text": "course evidence"},
                "chunk": "course evidence"}

    def test_singular_visual_question_analyzes_only_best_available_image(self):
        self.corpus.retrieve.return_value = [self.hit(n, image=n > 3) for n in range(1, 8)]
        result = main.ask({"question": "What does the flow diagram mean?"})
        self.assertEqual(self.vision.call_count, 1)
        for call in self.vision.call_args_list:
            prompt, paths = call.args
            self.assertEqual(len(paths), 1)
            number = int(Path(paths[0]).stem.split("-")[-1])
            self.assertIn("What does the flow diagram mean?", prompt)
            self.assertIn(f"[lecture.pptx p.{number}]", prompt)
        self.assertEqual([v for v in result["vision_sources"] if v["status"] == "success"], [
            {"file": "lecture.pptx", "page": n, "status": "success",
             "notes": f"Observed slide-{n}.png"} for n in (4,)])
        for n in (4,):
            self.assertIn(f"[lecture.pptx p.{n}]", result["vision_notes"])
            self.assertIn(f"Observed slide-{n}.png", result["vision_notes"])
        self.assertIn(result["vision_notes"], self.chat.call_args.args[0][1]["content"])

    def test_citations_label_slide_or_page_and_omit_missing_image_urls(self):
        slide = self.hit(2)["page"]
        pdf = self.hit(3, kind="pdf", file="reading.pdf")["page"]
        self.assertEqual(main._cite(slide)["kind"], "pptx")
        self.assertEqual(main._cite(slide)["label"], "Slide 2")
        self.assertEqual(main._cite(pdf)["label"], "Page 3")
        self.assertEqual(main._cite(slide)["image"], "/pages/slide-2.png")
        (self.index / slide["image"]).unlink()
        self.assertIsNone(main._cite(slide)["image"])
        pdf["image"] = "pages"
        self.assertIsNone(main._cite(pdf)["image"])

    def test_failed_or_missing_images_explicitly_limit_text_only_answer(self):
        missing = self.hit(1)
        (self.index / missing["page"]["image"]).unlink()
        self.corpus.retrieve.return_value = [missing, self.hit(2), self.hit(3, image=False)]
        self.vision.side_effect = RuntimeError("model unavailable")
        result = main.ask({"question": "Explain the chart"})
        self.assertTrue(result["answer"].startswith("Grounded answer"))
        self.assertEqual(result["vision_notes"], "")
        self.assertEqual(result["vision_sources"], [
            {"file": "lecture.pptx", "page": n, "status": status, "notes": ""}
            for n, status in ((1, "unavailable"), (2, "failed"), (3, "unavailable"))])
        warnings = "\n".join(result["visual_warnings"])
        self.assertIn("[lecture.pptx p.1]", warnings)
        self.assertIn("[lecture.pptx p.2]", warnings)
        self.assertIn("[lecture.pptx p.3]", warnings)
        self.assertIn("unavailable", warnings.lower())
        self.assertIn("failed", warnings.lower())
        self.assertIn("Visual limitations:", result["answer"])
        messages = self.chat.call_args.args[0]
        self.assertIn(warnings, messages[1]["content"])
        self.assertIn("Never claim", messages[0]["content"])
        self.assertIsNone(result["sources"][0]["image"])
        self.assertIsNone(result["sources"][2]["image"])

    def test_visual_prompts_require_observation_not_invented_labels_or_numbers(self):
        self.corpus.retrieve.return_value = [self.hit(1)]
        result = main.ask({"question": "Interpret this diagram"})
        prompt = self.vision.call_args.args[0]
        self.assertIn("observation", prompt.lower())
        self.assertIn("interpretation", prompt.lower())
        self.assertIn("unreadable labels", prompt.lower())
        self.assertIn("numbers", prompt.lower())
        self.assertIn("uncertain or unreadable", prompt.lower())
        system = self.chat.call_args.args[0][0]["content"]
        self.assertIn("observation", system.lower())
        self.assertIn("interpretation", system.lower())
        self.assertIn("automatically displays", system.lower())
        review_system = self.ground.call_args.args[0][0]["content"]
        self.assertIn("strict evidence editor", review_system.lower())
        self.assertIn("never say that you cannot embed", review_system.lower())
        self.assertEqual(result["visual_warnings"], [])
        self.assertEqual(result["answer"], "Grounded answer")

    def test_successful_visual_observations_are_course_evidence_not_text_only(self):
        hit = self.hit(7)
        hit['chunk'] = 'Benchmark Evaluation'
        self.corpus.retrieve.return_value = [hit]
        self.vision.side_effect = None
        self.vision.return_value = 'Two charts: Intelligence (higher is better), Cost per Task (lower is better).'
        main.ask({'question': 'Explain the two charts on slide 7.'})
        messages = self.chat.call_args.args[0]
        self.assertIn('Successful image observations are course evidence', messages[0]['content'])
        self.assertIn('Do not say visual information is absent just because extracted text omits it', messages[0]['content'])
        evidence = messages[1]['content']
        self.assertIn('COURSE EVIDENCE', evidence)
        self.assertLess(evidence.index('COURSE EVIDENCE'), evidence.index(self.vision.return_value))
        self.assertIn('exact names, labels, positions, spatial relationships, or data points only when they are clearly legible',
                      self.vision.call_args.args[0])

    def test_full_slide_layout_and_native_picture_detail_are_analyzed_together(self):
        hit = self.hit(7)
        self.corpus.retrieve.return_value = [hit]
        full = str(self.index / hit['page']['image'])
        detail = 'data:image/png;base64,bmF0aXZlLWZpeHR1cmU='
        with patch.object(main, '_slide_vision_images', return_value=[full, detail]):
            main.ask({'question':'Explain the chart on slide 7.'})
        self.assertEqual(self.vision.call_args.args[1], [full, detail])
        self.assertIn('first image is the complete', self.vision.call_args.args[0].lower())

    def test_no_hits_returns_empty_visual_metadata_without_model_calls(self):
        self.corpus.retrieve.return_value = []
        result = main.ask({"question": "Explain this diagram"})
        self.assertEqual(result.get("vision_sources"), [])
        self.assertEqual(result.get("visual_warnings"), [])
        self.assertEqual(result["vision_notes"], "")
        self.assertEqual(result["sources"], [])
        self.vision.assert_not_called()
        self.chat.assert_not_called()
        self.ground.assert_not_called()

    def test_empty_visual_response_is_a_failed_analysis(self):
        self.corpus.retrieve.return_value = [self.hit(1)]
        self.vision.side_effect = None
        self.vision.return_value = "   "
        result = main.ask({"question": "Explain the chart"})
        self.assertEqual(result["vision_sources"][0]["status"], "failed")
        self.assertEqual(result["vision_notes"], "")
        self.assertTrue(result["visual_warnings"])
        self.assertIn("Visual limitations:", result["answer"])

    def test_comparison_visual_calls_run_concurrently_and_keep_partial_success_attributed(self):
        self.corpus.retrieve.return_value = [self.hit(n) for n in range(1, 5)]
        barrier = threading.Barrier(2, timeout=2)

        def concurrent_description(prompt, paths):
            barrier.wait()
            if Path(paths[0]).name == "slide-2.png":
                raise RuntimeError("one model call failed")
            return self.describe(prompt, paths)

        self.vision.side_effect = concurrent_description
        result = main.ask({"question": "Explain the diagrams"})
        self.assertEqual(self.vision.call_count, 2)
        self.assertEqual([v["status"] for v in result["vision_sources"]],
                         ["success", "failed"])
        self.assertIn("[lecture.pptx p.1]", result["vision_notes"])
        self.assertNotIn("[lecture.pptx p.2]", result["vision_notes"])
        self.assertEqual(len(result["visual_warnings"]), 1)
        self.assertIn("[lecture.pptx p.2]", result["visual_warnings"][0])

    def test_limitation_notices_do_not_count_as_answer_evidence(self):
        self.corpus.retrieve.return_value = [self.hit(1, image=False)]
        result = main.ask({"question": "Explain the missing chart"})
        self.assertEqual(result["validation"]["sources_used"], [])
        self.assertFalse(result["validation"]["all_sources_supported"])

    def test_requirement_2_sources_keep_exact_week2_slide_and_image_url(self):
        filename = "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx"
        cases = [
            (33, 'Find the meme about Vibe Coding on "Prod".'),
            (7, "Explain the Benchmark Evaluation charts."),
        ]
        for slide, question in cases:
            with self.subTest(slide=slide):
                self.vision.reset_mock()
                self.chat.reset_mock()
                self.ground.reset_mock()
                hit = self.hit(slide, file=filename)
                self.corpus.retrieve.return_value = [hit]
                answer = f"Supported answer [{filename} p.{slide}]"
                self.chat.return_value = answer
                self.ground.return_value = answer
                result = main.ask({"question": question})
                self.assertEqual(len(result["sources"]), 1)
                self.assertEqual(result["sources"][0]["file"], filename)
                self.assertEqual(result["sources"][0]["page"], slide)
                self.assertEqual(result["sources"][0]["label"], f"Slide {slide}")
                self.assertEqual(result["sources"][0]["image"],
                                 f"/pages/slide-{slide}.png")
                self.assertTrue(result["validation"]["all_sources_supported"])

    def test_meme_locator_displays_only_slide_33_even_with_extra_candidates(self):
        filename = "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx"
        self.corpus.retrieve.return_value = [
            self.hit(33, file=filename),
            self.hit(27, file=filename),
            self.hit(35, file=filename),
            self.hit(31, file=filename),
        ]
        answer = (
            f'Direct observations: Slide 33 contains the Vibe Coding on "Prod" meme '
            f'[{filename} p.33]. Other retrieved candidates are not the requested meme '
            f'[{filename} p.27] [{filename} p.35].'
        )
        self.chat.return_value = answer
        self.ground.return_value = answer

        result = main.ask({"question": 'Find the meme about Vibe Coding on "Prod".'})

        self.assertEqual(
            [(source["file"], source["page"]) for source in result["sources"]],
            [(filename, 33)],
        )
        self.assertEqual(result["sources"][0]["image"], "/pages/slide-33.png")
        self.assertEqual(result["validation"]["sources_used"],
                         [{"file": filename, "page": 33}])
        self.assertTrue(result["validation"]["all_sources_supported"])

    def test_multi_source_visual_question_keeps_all_cited_support(self):
        filename = "lecture.pptx"
        self.corpus.retrieve.return_value = [self.hit(7), self.hit(8), self.hit(9)]
        answer = (f"The two charts can be compared [{filename} p.7] "
                  f"[{filename} p.8].")
        self.chat.return_value = answer
        self.ground.return_value = answer

        result = main.ask({"question": "Compare the charts on slides 7 and 8."})

        self.assertEqual([source["page"] for source in result["sources"]], [7, 8])

    def test_unsupported_visual_claims_are_revised_before_return(self):
        filename = "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx"
        self.corpus.retrieve.return_value = [self.hit(7, file=filename)]
        unsupported = f"The top chart proves Model Z costs $99 [{filename} p.7]"
        revised = (
            "Direct observations: The left chart is Intelligence and the right chart "
            f"is Cost per Task.\n\nInterpretation: The slide compares performance and cost "
            f"[{filename} p.7]"
        )
        self.chat.return_value = unsupported
        self.ground.return_value = revised
        result = main.ask({"question": "Explain slide 7"})
        self.assertEqual(result["answer"], revised)
        self.assertNotIn("Model Z", result["answer"])
        self.assertNotIn("$99", result["answer"])
        self.assertEqual(result["validation"]["grounding_review"]["outcome"],
                         "revised")
        review = self.ground.call_args.args[0][1]["content"]
        self.assertIn(unsupported, review)
        self.assertIn("ATTRIBUTED VISUAL OBSERVATIONS", review)

    def test_wrong_slide_number_does_not_pass_citation_validation(self):
        filename = "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx"
        self.corpus.retrieve.return_value = [self.hit(33, file=filename)]
        wrong = f"Meme explanation [{filename} p.7]"
        self.chat.return_value = wrong
        self.ground.return_value = wrong
        result = main.ask({"question": "Explain the meme on slide 33"})
        self.assertEqual(result["validation"]["sources_used"], [])
        self.assertEqual(result["validation"]["sources_not_cited"],
                         [{"file": filename, "page": 33}])
        self.assertFalse(result["validation"]["all_sources_supported"])

    def test_failed_grounding_review_rejects_unvalidated_draft(self):
        filename = "MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx"
        self.corpus.retrieve.return_value = [self.hit(33, file=filename)]
        self.chat.return_value = f"Unsupported celebrity claim [{filename} p.33]"
        self.ground.side_effect = RuntimeError("validator unavailable")
        result = main.ask({"question": "Explain the meme on slide 33"})
        self.assertNotIn("Unsupported celebrity claim", result["answer"])
        self.assertIn("could not be validated", result["answer"])
        self.assertEqual(result["validation"]["grounding_review"],
                         {"checked": True, "outcome": "rejected"})


if __name__ == "__main__":
    unittest.main()
