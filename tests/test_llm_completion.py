"""Final-answer-only contract for reasoning model endpoints."""
import unittest
from unittest.mock import Mock, patch
from app import llm


class ModelCompletionTests(unittest.TestCase):
    def response(self, content, reasoning=None, finish='stop'):
        r=Mock()
        r.json.return_value={'choices':[{'finish_reason':finish,'message':{'content':content,'reasoning':reasoning}}]}
        return r

    def test_requests_non_thinking_completed_answers_with_sufficient_budget(self):
        with patch.object(llm.requests,'post',return_value=self.response('Completed answer.','Internal thoughts')) as post:
            self.assertEqual(llm.chat([]),'Completed answer.')
            payload=post.call_args.kwargs['json']
            self.assertFalse(payload.get('chat_template_kwargs',{}).get('enable_thinking',True))
            self.assertGreaterEqual(payload['max_tokens'],4096)

    def test_truncated_content_is_not_presented_as_complete(self):
        with patch.object(llm.requests,'post',return_value=self.response('Partial answer',finish='length')):
            with self.assertRaisesRegex(RuntimeError,'final answer'):
                llm.chat([])

    def test_grounding_review_uses_deterministic_temperature(self):
        with patch.object(llm.requests, 'post',
                          return_value=self.response('Validated answer.')) as post:
            self.assertEqual(llm.ground([]), 'Validated answer.')
            self.assertEqual(post.call_args.kwargs['json']['temperature'], 0.0)

    def test_internal_reasoning_is_never_a_final_answer(self):
        with patch.object(llm.requests,'post',return_value=self.response(None,'We need to analyze the slide.',finish='length')):
            with self.assertRaisesRegex(RuntimeError,'final answer'):
                llm._complete('https://model.invalid/v1','test-key','fixture',[])


if __name__=='__main__':
    unittest.main()
