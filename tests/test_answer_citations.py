"""Presentation-only citations: keep provenance without repeated long filenames."""
from playwright.sync_api import expect
from test_qa_feedback import page

FILE='MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx'
SOURCE={'doc':'week-2','file':FILE,'kind':'pptx','page':33,'image':None}


def test_repeated_paragraph_citations_are_one_clickable_marker(page):
    cite=f'[{FILE} p.33]'
    answer=f'Direct observations:\nThe slide title is Vibe Coding {cite}. The main visual is Boromir {cite}. The caption warns about production {cite}.\n\nInterpretation:\nThis is a production warning {cite}.'
    page.route('**/api/ask',lambda r:r.fulfill(json={'answer':answer,'sources':[SOURCE]}))
    page.locator('#q').fill('Explain slide 33')
    page.locator('#askBtn').click()
    expect(page.locator('#answer')).to_contain_text('The main visual is Boromir')
    assert FILE not in page.locator('#answer').inner_text()
    expect(page.locator('.observation .citation-ref')).to_have_count(1)
    expect(page.locator('.interpretation .citation-ref')).to_have_count(1)
    link=page.locator('.observation .citation-ref')
    expect(link).to_have_text('[1]')
    expect(link).to_have_attribute('title',f'{FILE} · Slide 33')
    link.click()
    expect(page.locator('#source-1')).to_be_focused()
    expect(page.locator('#source-1')).to_contain_text(FILE)


def test_italics_and_bold_render_without_executing_html(page):
    page.evaluate('s=>renderAnswer(s)', 'Boromir from *The Lord of the Rings*. **Production warning**. <img src=x onerror="window.pwned=true">')
    expect(page.locator('#answer em')).to_have_text('The Lord of the Rings')
    expect(page.locator('#answer strong')).to_have_text('Production warning')
    expect(page.locator('#answer img')).to_have_count(0)
    assert page.evaluate('window.pwned') is None


def test_multiple_sources_and_unmatched_references_are_not_conflated(page):
    second={**SOURCE,'file':'Week 5.pptx','doc':'week-5'}
    answer=f'Comparison [{FILE} p.33] versus [Week 5.pptx p.33]. Unknown [Missing.pdf p.2].'
    page.evaluate('d=>{renderAnswer(d.answer,d.sources);renderSources(d.sources)}',{'answer':answer,'sources':[SOURCE,second]})
    expect(page.locator('#answer .citation-ref')).to_have_count(2)
    expect(page.locator('#answer .citation-ref').nth(0)).to_have_attribute('href','#source-1')
    expect(page.locator('#answer .citation-ref').nth(1)).to_have_attribute('href','#source-2')
    expect(page.locator('#answer')).to_contain_text('[Missing.pdf p.2]')
