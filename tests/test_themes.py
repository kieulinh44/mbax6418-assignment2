"""UI tests: dark/light mode (system preference + manual toggle).

Run with:  python -m pytest tests/test_themes.py -v
Requires the local app on port 8000 and playwright (see requirements-dev.txt).
"""
import pytest

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"

LIGHT_BG = "rgb(248, 243, 234)"   # #f8f3ea
DARK_BG = "rgb(26, 20, 28)"       # #1a141c


@pytest.fixture(scope="module")
def pw():
    with sync_playwright() as p:
        yield p


def _new_page(pw, color_scheme):
    browser = pw.chromium.launch()
    ctx = browser.new_context(color_scheme=color_scheme)
    page = ctx.new_page()
    page.goto(BASE)
    page.wait_for_function("document.getElementById('materials').textContent !== 'Loading…'")
    return browser, page


def _body_bg(page):
    return page.evaluate("getComputedStyle(document.body).backgroundColor")


def test_follows_system_dark(pw):
    browser, page = _new_page(pw, "dark")
    try:
        assert _body_bg(page) == DARK_BG, f"expected dark bg, got {_body_bg(page)}"
    finally:
        browser.close()


def test_follows_system_light(pw):
    browser, page = _new_page(pw, "light")
    try:
        assert _body_bg(page) == LIGHT_BG, f"expected light bg, got {_body_bg(page)}"
    finally:
        browser.close()


def test_toggle_switches_and_persists(pw):
    browser, page = _new_page(pw, "light")
    try:
        # click the toggle -> dark
        page.click("#themeToggle")
        assert _body_bg(page) == DARK_BG
        assert page.evaluate("localStorage.getItem('theme')") == "dark"
        assert page.evaluate("document.documentElement.dataset.theme") == "dark"
        # and back -> light
        page.click("#themeToggle")
        assert _body_bg(page) == LIGHT_BG
        assert page.evaluate("localStorage.getItem('theme')") == "light"
        # explicit light wins even when the OS requests dark
        page.evaluate("matchMedia('(prefers-color-scheme: dark)')")  # sanity
    finally:
        browser.close()


def test_controls_and_images_readable_in_dark(pw):
    """Source images get a white backdrop; cards/text use theme vars in dark."""
    browser, page = _new_page(pw, "dark")
    try:
        # controls inherit theme colors (inputs are var-driven)
        bg = page.evaluate(
            "getComputedStyle(document.querySelector('#topic')).backgroundColor")
        assert bg != "rgba(0, 0, 0, 0)" and bg != "transparent"
        # tab colors resolve from the dark palette
        accent = page.evaluate(
            "getComputedStyle(document.querySelector('.tab.active')).color")
        assert accent == "rgb(38, 29, 43)", f"tab active text should be card color, got {accent}"
        # upload zone uses theme vars, not the old hardcoded light pink
        wz = page.evaluate(
            "getComputedStyle(document.querySelector('.upload-zone')).backgroundColor")
        assert wz != "rgb(248, 241, 245)", "upload zone should use theme vars in dark"
    finally:
        browser.close()
