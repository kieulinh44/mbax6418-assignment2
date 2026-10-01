"""Browser regression tests. Run with .venv/bin/python -m unittest discover -s tests -v.
Requires the local app on port 8000 and playwright (dev dependency).

Skipped (not failed) when playwright isn't installed, so plain `pytest -q`
collection always works — including CI with only the light test deps.
"""
import unittest

try:
    from playwright.sync_api import sync_playwright
    _NO_PLAYWRIGHT = None
except ImportError:  # pragma: no cover
    sync_playwright = None
    _NO_PLAYWRIGHT = "playwright not installed (pip install playwright && playwright install chromium)"


@unittest.skipIf(_NO_PLAYWRIGHT is not None, _NO_PLAYWRIGHT)
class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch()
        self.page = self.browser.new_page()
        self.page.goto('http://127.0.0.1:8000')
        self.page.wait_for_function("document.getElementById('files').textContent !== 'Loading…'")

    def tearDown(self):
        self.browser.close()
        self.pw.stop()

    def test_uploads_only_visible_in_materials_tab(self):
        self.assertEqual(self.page.locator('[data-tab="materials"]').count(), 1)
        self.assertFalse(self.page.locator('#uploadInput').is_visible())
        self.page.locator('[data-tab="materials"]').click()
        self.assertTrue(self.page.locator('#uploadInput').is_visible())
        self.assertTrue(self.page.locator('#files').is_visible())
        self.page.locator('[data-tab="quiz"]').click()
        self.assertFalse(self.page.locator('#uploadInput').is_visible())
        self.assertFalse(self.page.locator('#files').is_visible())
        self.page.locator('[data-tab="qa"]').click()
        self.assertTrue(self.page.locator('#q').is_visible())
        self.assertFalse(self.page.locator('#uploadInput').is_visible())

    def test_contributor_credit_and_responsive_theme(self):
        credit = 'Contributed by Linh Nguyen, Jamie Lin, Kelly Nguyen, Jana Chittarath, and Mirina Gurung'
        self.assertIn(credit, self.page.locator('header').inner_text())
        heading_size = self.page.locator('h1').evaluate('(el)=>parseFloat(getComputedStyle(el).fontSize)')
        credit_size = self.page.locator('.contributors').evaluate('(el)=>parseFloat(getComputedStyle(el).fontSize)')
        self.assertLess(credit_size, heading_size)
        self.assertIn('DM Sans', self.page.locator('body').evaluate('(el)=>getComputedStyle(el).fontFamily'))
        for width in [390, 768, 1440]:
            self.page.set_viewport_size({'width': width, 'height': 900})
            for tab in ['qa', 'quiz', 'materials']:
                self.page.locator(f'[data-tab="{tab}"]').click()
                self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))

    def test_tab_deep_links_open_requested_panel(self):
        for tab in ['qa', 'quiz', 'materials']:
            self.page.goto(f'http://127.0.0.1:8000/?tab={tab}')
            self.assertEqual(self.page.locator('.tab.active').get_attribute('data-tab'), tab)
            self.assertTrue(self.page.locator(f'#panel-{tab}').is_visible())

    def test_check_answers_submits_selected_answers(self):
        payloads = []
        self.page.route('**/api/quiz', lambda route: route.fulfill(json={
            'quiz_id': 'fixture', 'title': 'Browser regression fixture', 'questions': [{
                'id': 'q1', 'question': 'Choose the first option.', 'options': ['First', 'Second'],
                'source': {'file': 'fixture.pdf', 'page': 1, 'excerpt': 'Fixture source'}
            }]
        }))
        def capture_grade(route):
            payloads.append(route.request.post_data_json)
            route.fulfill(json={'results': []})
        self.page.route('**/api/quiz/grade', capture_grade)
        self.page.locator('[data-tab="quiz"]').click()
        self.page.locator('#genBtn').click()
        self.page.locator('.question input').first.check()
        self.page.locator('#gradeBtn').click()
        self.page.wait_for_function("document.getElementById('quizres').textContent.length > 0")
        self.assertEqual(payloads[0]['answers'], {'q1': 0})

    def test_upload_confirmation_lists_each_filename_once(self):
        # API fixture reproduces the page-level added list returned by ingestion.
        self.page.route('**/api/upload', lambda route: route.fulfill(json={
            'ok': True, 'result': {'added': ['lecture.pdf'] * 12, 'pages': 15}
        }))
        self.page.locator('[data-tab="materials"]').click()
        self.page.locator('#uploadInput').set_input_files({
            'name': 'lecture.pdf', 'mimeType': 'application/pdf', 'buffer': b'UI test fixture'
        })
        self.page.locator('#uploadBtn').click()
        self.page.wait_for_function("document.getElementById('uploadStatus').textContent.includes('Indexed')")
        self.assertEqual(self.page.locator('#uploadStatus').inner_text().count('lecture.pdf'), 1)


if __name__ == '__main__':
    unittest.main()
