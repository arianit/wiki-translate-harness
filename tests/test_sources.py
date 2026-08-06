from pathlib import Path

from wiki_translate_harness.sources import ArticleInput, parse_source_ref, resolve_static_inputs


def test_parse_lang_prefix():
    lang, title = parse_source_ref("sq:Gjergj Arianiti")
    assert lang == "sq"
    assert title == "Gjergj Arianiti"


def test_parse_lang_prefix_with_region_subtag():
    lang, title = parse_source_ref("zh-hans:Some Title")
    assert lang == "zh-hans"
    assert title == "Some Title"


def test_parse_full_url_with_scheme():
    lang, title = parse_source_ref("https://sq.wikipedia.org/wiki/Gjergj_Arianiti")
    assert lang == "sq"
    assert title == "Gjergj Arianiti"


def test_parse_full_url_without_scheme():
    lang, title = parse_source_ref("en.wikipedia.org/wiki/Paris")
    assert lang == "en"
    assert title == "Paris"


def test_parse_mobile_subdomain_url():
    lang, title = parse_source_ref("https://sq.m.wikipedia.org/wiki/Gjergj_Arianiti")
    assert lang == "sq"
    assert title == "Gjergj Arianiti"


def test_parse_url_strips_query_and_fragment():
    lang, title = parse_source_ref("https://en.wikipedia.org/wiki/Paris?action=history#History")
    assert lang == "en"
    assert title == "Paris"


def test_parse_url_decodes_percent_encoding():
    lang, title = parse_source_ref("https://sr.wikipedia.org/wiki/%D0%9D%D0%B8%D1%88")
    assert lang == "sr"
    assert title == "Ниш"


def test_no_prefix_returns_none_lang():
    lang, title = parse_source_ref("Paris")
    assert lang is None
    assert title == "Paris"


def test_namespace_prefix_not_mistaken_for_lang():
    # "Category" is capitalized -> real MediaWiki namespace, not a lang code
    lang, title = parse_source_ref("Category:Physics")
    assert lang is None
    assert title == "Category:Physics"


def test_namespace_prefix_template_not_mistaken_for_lang():
    lang, title = parse_source_ref("Template:Infobox royalty")
    assert lang is None
    assert title == "Template:Infobox royalty"


def test_resolve_static_inputs_applies_lang_prefix_to_title():
    inputs = resolve_static_inputs(title="sq:Gjergj Arianiti", titles_file=None, file=None, directory=None)
    assert len(inputs) == 1
    assert inputs[0].title == "Gjergj Arianiti"
    assert inputs[0].source_lang == "sq"


def test_resolve_static_inputs_mixed_titles_file(tmp_path: Path):
    titles_file = tmp_path / "titles.txt"
    titles_file.write_text("Paris\nsq:Gjergj Arianiti\n# a comment\nsr:Ниш\n")
    inputs = resolve_static_inputs(title=None, titles_file=titles_file, file=None, directory=None)
    assert [(i.title, i.source_lang) for i in inputs] == [
        ("Paris", None),
        ("Gjergj Arianiti", "sq"),
        ("Ниш", "sr"),
    ]


def test_resolve_static_inputs_local_file_has_no_lang(tmp_path: Path):
    wiki_file = tmp_path / "Some_Article.wiki"
    wiki_file.write_text("content")
    inputs = resolve_static_inputs(title=None, titles_file=None, file=wiki_file, directory=None)
    assert inputs[0].source_lang is None
    assert inputs[0].title == "Some Article"


def test_article_input_default_source_lang_is_none():
    item = ArticleInput(title="Paris")
    assert item.source_lang is None
