"""Offline UI regressions: no server, model, upload, or deletion requests.

Serve the actual frontend through Playwright interception; every other request
is explicitly fulfilled or aborted. Failures are the RED tests for UI fixes.
"""
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

HTML = Path(__file__).resolve().parents[1] / 'static' / 'index.html'
QUESTION = 'Which approach retrieves course evidence?'
QUIZ = {
    'quiz_id': 'accessibility-fixture', 'title': 'Evidence practice',
    'questions': [{
        'id': 'q1', 'question': QUESTION, 'options': ['RAG', 'Guessing'],
        'source': {'file': 'fixture.pdf', 'page': 1, 'excerpt': 'RAG retrieves evidence.'},
    }],
}
GRADE = {
    'results': [{'id': 'q1', 'correct_answer': 0, 'your_answer': 0,
                 'correct': True, 'explanation': 'RAG retrieves evidence.'}],
    'score': 1, 'total': 1, 'percent': 100,
}


@pytest.fixture
def page():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(reduced_motion='reduce')
        def intercept(route):
            path = route.request.url.split('://', 1)[-1].partition('/')[2]
            if route.request.url == 'http://arcade.test/':
                route.fulfill(content_type='text/html', body=HTML.read_text())
            elif path == 'api/materials':
                route.fulfill(json={'materials': []})
            elif path == 'api/files':
                route.fulfill(json={'documents': []})
            elif path == 'api/quiz':
                route.fulfill(json=QUIZ)
            elif path == 'api/quiz/grade':
                route.fulfill(json=GRADE)
            else:
                route.abort()
        page.route('**/*', intercept)
        page.goto('http://arcade.test/')
        yield page
        browser.close()


def open_quiz(page):
    page.locator('[data-tab="quiz"]').click()
    page.locator('#genBtn').click()
    expect(page.locator('.question')).to_have_count(1)


def test_quiz_radio_group_is_named_by_its_question(page):
    """Individual option labels do not identify their shared question."""
    open_quiz(page)
    groups = page.get_by_role('group', name=QUESTION).or_(
        page.get_by_role('radiogroup', name=QUESTION))
    expect(groups).to_have_count(1)
    expect(groups.get_by_role('radio')).to_have_count(2)


def test_grading_failure_is_visible_and_can_be_retried(page):
    """A failed HTTP response must not become an unhandled JS exception."""
    open_quiz(page)
    page.get_by_role('radio', name='A. RAG').check()
    attempts = []
    def grade(route):
        attempts.append(route.request.post_data_json)
        if len(attempts) == 1:
            route.fulfill(status=503, json={'detail': 'Grading temporarily unavailable'})
        else:
            route.fulfill(json=GRADE)
    page.route('**/api/quiz/grade', grade)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.locator('#gradeBtn').click()
    expect(page.locator('#panel-quiz')).to_contain_text('Grading temporarily unavailable')
    expect(page.locator('#gradeBtn')).to_be_enabled()
    expect(page.get_by_role('radio', name='A. RAG')).to_be_checked()
    page.locator('#gradeBtn').click()
    expect(page.locator('#quizres')).to_contain_text('100%')
    assert len(attempts) == 2
    assert not errors, errors


def test_grading_respects_reduced_motion_preference(page):
    """CSS scroll-behavior does not override explicit JS smooth scrolling."""
    open_quiz(page)
    page.get_by_role('radio', name='A. RAG').check()
    page.evaluate('''() => {
        window.requestedScrolls = [];
        const original = window.scrollTo.bind(window);
        window.scrollTo = (...args) => {
            window.requestedScrolls.push(args);
            return original(...args);
        };
    }''')
    page.locator('#gradeBtn').click()
    expect(page.locator('#quizres')).to_contain_text('100%')
    smooth = page.evaluate('''window.requestedScrolls.some(args =>
        args[0] && args[0].behavior === 'smooth')''')
    assert not smooth, 'Grading explicitly requested smooth scrolling with reduced motion enabled'
