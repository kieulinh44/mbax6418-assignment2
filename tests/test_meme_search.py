"""Exhaustive visual collection search contracts; model I/O is mocked."""
import importlib
import json
from unittest.mock import Mock
import pytest
from app import config, hybrid, llm


def module():
    return importlib.import_module('app.visual_search')


@pytest.fixture
def pages(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DATA_INDEX',str(tmp_path))
    rows=[]
    for n in range(1,10):
        path=tmp_path/f'slide-{n}.png';path.write_bytes(f'image-{n}'.encode())
        rows.append({'doc':'week-2','file':'Week 2.pptx','page':n,'kind':'pptx',
                     'text':'Ordinary slide title','image':path.name,'section':None})
    rows.append({'doc':'week-5','file':'Week 5.pptx','page':1,'kind':'pptx',
                 'text':'meme','image':'slide-1.png'})
    return rows


def test_collection_respects_explicit_singular_slide(pages,monkeypatch):
    classify=Mock(return_value=json.dumps({'match':True,'description':'Visible caption joke'}))
    monkeypatch.setattr(llm,'vision',classify)
    result=module().search_memes('show all memes on slide 7 from week 2',pages)
    assert [p['page'] for p in result['matches']]==[7]
    assert result['coverage']['total']==1
    assert classify.call_count==1


def test_collection_search_inspects_whole_deck_not_top_six(pages,monkeypatch):
    # The seventh/ninth slides have no word "meme" in their extracted text.
    classify=Mock(side_effect=lambda prompt,images,**kwargs:json.dumps({
        'match':images[0].endswith(('slide-7.png','slide-9.png')),
        'description':'Visible caption joke' if images[0].endswith(('slide-7.png','slide-9.png')) else 'Informational slide'}))
    monkeypatch.setattr(llm,'vision',classify)
    result=module().search_memes('find all the memes from the week 2 slides',pages)
    assert [(p['file'],p['page']) for p in result['matches']]==[('Week 2.pptx',7),('Week 2.pptx',9)]
    assert classify.call_count==9
    assert result['coverage']=={'total':9,'reviewed':9,'failed':0,'uncertain':0,'complete':True}


def test_failed_image_analysis_never_claims_complete_collection(pages,monkeypatch):
    def classify(prompt,images,**kwargs):
        if images[0].endswith('slide-9.png'):raise RuntimeError('offline')
        return json.dumps({'match':False,'description':'No joke'})
    monkeypatch.setattr(llm,'vision',classify)
    result=module().search_memes('all memes week 2',pages)
    assert result['coverage']['reviewed']==8
    assert result['coverage']['failed']==1
    assert result['coverage']['complete'] is False
    assert result['matches']==[]


def test_uncertain_and_invalid_classifications_are_not_negative_evidence(pages,monkeypatch):
    monkeypatch.setattr(llm,'vision',lambda *a,**k:json.dumps({'match':None,'description':'Too small to tell'}))
    result=module().search_memes('all memes week 2',pages)
    assert result['coverage']['uncertain']==9
    assert result['coverage']['complete'] is False
    assert result['matches']==[]
    # Invalid responses must not be cached as true/false.
    monkeypatch.setattr(llm,'vision',lambda *a,**k:json.dumps({'match':'false','description':'wrong type'}))
    result=module().search_memes('all memes week 2',pages)
    assert result['coverage']['failed']==9


def test_cache_uses_image_content_not_just_filename(pages,monkeypatch,tmp_path):
    classify=Mock(return_value=json.dumps({'match':False,'description':'No meme'}))
    monkeypatch.setattr(llm,'vision',classify)
    module().search_memes('all memes week 2',pages)
    module().search_memes('all memes week 2',pages)
    assert classify.call_count==9
    (tmp_path/'slide-3.png').write_bytes(b'new image content')
    module().search_memes('all memes week 2',pages)
    assert classify.call_count==10


def test_conflicting_document_filter_does_not_search_wrong_week(pages,monkeypatch):
    classify=Mock();monkeypatch.setattr(llm,'vision',classify)
    result=module().search_memes('all memes week 2',pages,doc='week-5')
    assert result['coverage']['total']==0
    assert result['matches']==[]
    classify.assert_not_called()


def test_ask_collection_bypasses_ranked_top_k_and_displays_only_confirmed_memes(monkeypatch):
    from app import main
    match={'doc':'week-2','file':'Week 2.pptx','page':9,'kind':'pptx',
           'text':'Ordinary title','image':None,'visual_description':'Visible caption joke'}
    monkeypatch.setattr(module(),'search_memes',lambda *a,**k:{
        'matches':[match], 'limitations':[], 'files':['Week 2.pptx'],
        'coverage':{'total':42,'reviewed':42,'failed':0,'uncertain':0,'complete':True}})
    monkeypatch.setattr(hybrid,'load_pages',lambda:[])
    monkeypatch.setattr(hybrid,'get_corpus',Mock(side_effect=AssertionError('Collection must not use ranked top-k')))
    response=main.ask({'question':'find all the memes from the week 2 slides'})
    assert [(p['file'],p['page']) for p in response['sources']]==[('Week 2.pptx',9)]
    assert response['search_coverage']['reviewed']==42
    assert '42' in response['answer']
    assert '[Week 2.pptx p.9]' in response['answer']


def test_meme_intent_does_not_hijack_explanation_or_comparison_queries():
    m=module()
    assert m.is_meme_collection('find all the memes from the week 2 slides')
    assert m.is_meme_collection('show memes in week 2')
    assert not m.is_meme_collection('Explain the meme on slide 33')
    assert not m.is_meme_collection('Find the meme about Vibe Coding on Prod')
    assert not m.is_meme_collection('Show me the meme on slide 33')
    assert not m.is_meme_collection('What is a meme?')
    assert not m.is_meme_collection('Compare memes and diagrams')
