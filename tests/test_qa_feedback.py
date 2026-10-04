"""Offline browser regressions for visible Q&A loading and result discovery."""
from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright, expect

HTML = Path(__file__).resolve().parents[1] / 'static/index.html'
SOURCE = {'doc':'week-2','file':'Week 2.pptx','kind':'pptx','page':33,
          'image':'/pages/test.png','excerpt':'Original slide'}
RESPONSE = {'answer':'Direct observations:\nA captioned meme.\n\nInterpretation:\nA production warning.', 'sources':[SOURCE]}

@pytest.fixture
def page():
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1366,'height':768},reduced_motion='reduce')
        page.route('**/*', lambda r:r.abort())
        page.route('http://qa.test/',lambda r:r.fulfill(body=HTML.read_text(),content_type='text/html'))
        page.route('**/api/materials',lambda r:r.fulfill(json={'materials':[]}))
        page.route('**/api/files',lambda r:r.fulfill(json={'documents':[{'doc':'week-2','file':'Week 2.pptx','kind':'pptx','pages':42}]}))
        page.goto('http://qa.test/')
        yield page
        browser.close()

@pytest.mark.parametrize('theme',['light','dark'])
def test_results_begin_inside_laptop_viewport_without_scroll(page,theme):
    page.evaluate('t=>applyTheme(t)',theme)
    page.route('**/api/ask',lambda r:r.fulfill(json=RESPONSE))
    page.locator('#q').fill('Explain sycophancy')
    page.locator('#askBtn').click()
    expect(page.locator('#answer')).to_contain_text('A captioned meme')
    bounds=page.locator('#result').bounding_box()
    assert page.evaluate('scrollY')==0
    assert bounds['y']+100 <= 768, bounds

@pytest.mark.parametrize('theme',['light','dark'])
def test_explanation_has_visible_local_and_viewport_loading_then_reveals_result(page,theme):
    page.evaluate('t=>applyTheme(t)',theme)
    page.evaluate('d=>{renderAnswer(d.answer);renderSources(d.sources);$("result").style.display="grid"}',RESPONSE)
    pending=[]
    page.route('**/api/ask',lambda r:pending.append(r))
    page.locator('.explain-slide').click()
    expect(page.locator('.explain-slide')).to_be_disabled()
    expect(page.locator('.explain-slide')).to_contain_text('Explaining')
    expect(page.locator('.explain-status')).to_contain_text('slide 33')
    expect(page.locator('#qaProgress')).to_be_visible()
    expect(page.locator('#qaProgress')).to_contain_text('Explaining slide 33')
    box=page.locator('#qaProgress').bounding_box()
    assert 0 <= box['y'] < 768 and box['y']+box['height']<=768
    expect(page.locator('#result')).to_have_attribute('aria-busy','true')
    page.wait_for_function('document.getElementById("askBtn").disabled')
    pending[0].fulfill(json=RESPONSE)
    expect(page.locator('#qaProgress')).to_be_hidden()
    expect(page.locator('.explain-slide')).to_be_enabled()
    expect(page.locator('#result')).to_have_attribute('aria-busy','false')
    expect(page.locator('#resultHeading')).to_be_focused()
    box=page.locator('#resultHeading').bounding_box()
    assert 0<=box['y']<768


def test_failed_explanation_clears_loading_and_retry_is_available(page):
    page.evaluate('d=>{renderSources(d.sources);$("result").style.display="grid"}',RESPONSE)
    page.route('**/api/ask',lambda r:r.fulfill(status=502,json={'detail':'Temporary model failure'}))
    page.locator('.explain-slide').click()
    expect(page.locator('#answer')).to_contain_text('Temporary model failure')
    expect(page.locator('#qaProgress')).to_be_hidden()
    expect(page.locator('#askBtn')).to_be_enabled()
    expect(page.locator('#resultHeading')).to_be_focused()
