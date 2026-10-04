"""Strategy Arcade interaction and accessibility regressions; local app required."""
import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture
def page():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(color_scheme='light')
        page.goto('http://127.0.0.1:8000')
        yield page
        browser.close()


def test_unchecked_quiz_radios_have_nontext_contrast(page):
    from test_themes import contrast
    page.emulate_media(reduced_motion='reduce')
    page.route('**/api/quiz',lambda route:route.fulfill(json={
        'quiz_id':'fixture','title':'Control contrast','questions':[
            {'id':'q1','question':'Choose a strategy','options':['Retrieve','Guess'],
             'source':{'file':'fixture.pdf','page':1,'excerpt':'Evidence'}}]}))
    page.locator('[data-tab="quiz"]').click()
    page.locator('#genBtn').click()
    page.locator('.opt input').first.wait_for()
    for mode in ['light','dark']:
        page.evaluate('(mode)=>document.documentElement.dataset.theme=mode',mode)
        colors=page.locator('.opt input').first.evaluate('''el=>{const s=getComputedStyle(el);return {outline:s.borderTopColor,inside:s.backgroundColor,outside:getComputedStyle(el.closest('.opt')).backgroundColor};}''')
        assert contrast(colors['outline'],colors['inside']) >= 3, (mode,colors)
        assert contrast(colors['outline'],colors['outside']) >= 3, (mode,colors)


def test_collection_request_explains_full_deck_scan_while_loading(page):
    pending=[]
    page.route('**/api/ask',lambda route:pending.append(route))
    page.locator('#q').fill('find all the memes from the week 2 slides')
    page.locator('#askBtn').click()
    try:
        assert 'Inspecting' in page.locator('#askStatus').inner_text()
        assert 'cached' in page.locator('#askStatus').inner_text()
        assert page.locator('#askBtn').is_disabled()
    finally:
        for route in pending:route.abort()


def test_no_ambiguous_initials_logo(page):
    assert page.locator('.brand-mark').count() == 0
    assert 'Strategy Arcade' in page.locator('header h1').inner_text()


def test_arcade_identity_and_working_prompt_suggestion(page):
    assert 'Strategy Arcade' in page.title()
    suggestion = page.locator('[data-question]').first
    assert suggestion.is_visible()
    expected = suggestion.get_attribute('data-question')
    suggestion.click()
    assert page.locator('#q').input_value() == expected
    assert page.locator('#q').evaluate('(el)=>el===document.activeElement')


def test_keyboard_navigation_between_workspaces(page):
    page.locator('[data-tab="qa"]').focus()
    page.keyboard.press('ArrowRight')
    assert page.locator('[data-tab="quiz"]').get_attribute('aria-selected') == 'true'
    assert page.locator('#panel-quiz').is_visible()
    page.keyboard.press('End')
    assert page.locator('#panel-materials').is_visible()
    page.keyboard.press('Home')
    assert page.locator('#panel-qa').is_visible()


@pytest.mark.parametrize('mode', ['light', 'dark'])
def test_responsive_controls_and_theme_persistence(page, mode):
    page.evaluate('(mode)=>localStorage.setItem("theme",mode)', mode)
    page.reload()
    assert page.locator('html').get_attribute('data-theme') == mode
    for width in [390, 768, 1440]:
        page.set_viewport_size({'width':width,'height':900})
        for tab in ['qa', 'quiz', 'materials']:
            page.locator(f'[data-tab="{tab}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        for selector in ['.tab', '#themeToggle', '#uploadBtn']:
            for button in page.locator(selector).all():
                if button.is_visible():
                    assert button.bounding_box()['height'] >= 44


def test_answers_render_untrusted_html_as_text(page):
    page.route('**/api/ask',lambda route:route.fulfill(json={
        'answer':'<img src=x onerror="window.injected=true">', 'sources':[], 'visual_warnings':[]}))
    page.locator('#q').fill('Test safe rendering')
    page.locator('#askBtn').click()
    page.wait_for_function('document.getElementById("answer").textContent.includes("onerror")')
    assert page.locator('#answer img').count() == 0
    assert page.evaluate('window.injected===undefined')


def test_reveal_submits_empty_answers_and_displays_explanations(page):
    payloads=[]
    page.route('**/api/quiz',lambda route:route.fulfill(json={
        'quiz_id':'fixture','title':'Business AI challenge','questions':[
            {'id':'q1','question':'Which approach retrieves context?', 'options':['RAG','Guessing'],
             'source':{'file':'fixture.pdf','page':1,'excerpt':'RAG retrieves relevant material.'}}]}))
    def grade(route):
        payloads.append(route.request.post_data_json)
        route.fulfill(json={'results':[{'id':'q1','correct_answer':0,'your_answer':None,
             'correct':False,'explanation':'RAG retrieves source context.'}], 'score':0,'total':1,'percent':0})
    page.route('**/api/quiz/grade',grade)
    page.locator('[data-tab="quiz"]').click()
    page.locator('#genBtn').click()
    page.locator('.question input').first.check()
    page.locator('#revealBtn').click()
    page.locator('#quizres .expl').first.wait_for()
    assert payloads[0]['answers'] == {}
    assert 'RAG retrieves source context.' in page.locator('#quizres').inner_text()
    assert page.locator('.opt.good').count() == 1
    assert page.locator('.question input').first.is_disabled()


def test_ask_failure_preserves_question_and_allows_retry(page):
    page.route('**/api/ask', lambda route: route.fulfill(status=502,json={'detail':'Test service unavailable'}))
    page.locator('#q').fill('Explain RAG for business applications')
    page.locator('#askBtn').click()
    page.wait_for_function('!document.getElementById("askBtn").disabled')
    assert page.locator('#q').input_value() == 'Explain RAG for business applications'
    assert 'Test service unavailable' in page.locator('#result').inner_text()
