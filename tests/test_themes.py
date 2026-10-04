"""Theme behavior and contrast regressions, independent of a specific palette."""
import re
import pytest
from playwright.sync_api import sync_playwright

BASE = 'http://127.0.0.1:8000'

@pytest.fixture(scope='module')
def pw():
    with sync_playwright() as p:
        yield p

def _new_page(pw, color_scheme):
    browser = pw.chromium.launch()
    page = browser.new_page(color_scheme=color_scheme)
    page.goto(BASE)
    page.wait_for_function("document.getElementById('materials').textContent !== 'Loading…'")
    return browser, page

def _body_bg(page):
    return page.evaluate('getComputedStyle(document.body).backgroundColor')

def luminance(rgb):
    channels = [float(n)/255 for n in re.findall(r'[\d.]+', rgb)[:3]]
    linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in channels]
    return sum(a*b for a,b in zip(linear,[.2126,.7152,.0722]))

def contrast(a,b):
    high,low = sorted([luminance(a),luminance(b)],reverse=True)
    return (high+.05)/(low+.05)

def test_follows_system_dark(pw):
    browser,page = _new_page(pw,'dark')
    try:
        assert luminance(_body_bg(page)) < .15
    finally:
        browser.close()

def test_follows_system_light(pw):
    browser,page = _new_page(pw,'light')
    try:
        assert luminance(_body_bg(page)) > .7
    finally:
        browser.close()

def test_toggle_switches_and_persists(pw):
    browser,page = _new_page(pw,'light')
    try:
        light = _body_bg(page)
        page.click('#themeToggle')
        dark = _body_bg(page)
        assert luminance(dark) < luminance(light)
        assert page.evaluate("localStorage.getItem('theme')") == 'dark'
        page.reload()
        assert _body_bg(page) == dark
        page.click('#themeToggle')
        assert _body_bg(page) == light
        assert page.evaluate("localStorage.getItem('theme')") == 'light'
        page.emulate_media(color_scheme='dark')
        assert _body_bg(page) == light
    finally:
        browser.close()

@pytest.mark.parametrize('mode',['light','dark'])
def test_controls_and_text_have_aa_contrast(pw,mode):
    browser,page = _new_page(pw,mode)
    try:
        for selector in ['body','#q','#topic','.tab.active','#askBtn','.study-note strong']:
            colors = page.locator(selector).evaluate('''el=>{
                const s=getComputedStyle(el);let bg=s.backgroundColor,p=el;
                while((bg==='rgba(0, 0, 0, 0)' || bg==='transparent') && p.parentElement){p=p.parentElement;bg=getComputedStyle(p).backgroundColor;}
                return [s.color,bg];
            }''')
            assert contrast(*colors) >= 4.5, (mode,selector,colors,contrast(*colors))
    finally:
        browser.close()
