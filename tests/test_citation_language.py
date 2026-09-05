import httpx
import pytest
import respx

from wiki_translation_harness.citation_language import (
    dedupe_short_footnotes,
    detect_source_language,
    fill_missing_citation_languages,
    fix_citation_param_names,
    guess_language_from_title,
)


def test_guess_language_from_title_too_short_returns_none():
    assert guess_language_from_title("Hi") is None


def test_guess_language_from_title_english():
    assert guess_language_from_title("This is a fairly normal English sentence used as a title") == "en"


def test_guess_language_from_title_rejects_low_confidence_short_phrase():
    # Regression: langdetect's detect() has no notion of "I don't know" and
    # confidently called this 9-character phrase Danish; detect_langs()
    # confidence-gating must reject it instead of writing a wrong language.
    assert guess_language_from_title("Some Page") is None


@pytest.mark.asyncio
async def test_detect_source_language_from_content_language_header():
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-language": "sr-Latn"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            lang = await detect_source_language(client, "https://example.com/page")
    assert lang == "sr"


@pytest.mark.asyncio
async def test_detect_source_language_from_html_lang_attribute():
    html = '<html lang="de"><head><title>x</title></head><body>hallo</body></html>'
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/html"}, text=html)
        )
        async with httpx.AsyncClient() as client:
            lang = await detect_source_language(client, "https://example.com/page")
    assert lang == "de"


@pytest.mark.asyncio
async def test_detect_source_language_falls_back_to_text_detection():
    html = (
        "<html><body><p>" + ("Dies ist ein deutscher Beispieltext mit vielen Wörtern. " * 10) + "</p></body></html>"
    )
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/html"}, text=html)
        )
        async with httpx.AsyncClient() as client:
            lang = await detect_source_language(client, "https://example.com/page")
    assert lang == "de"


@pytest.mark.asyncio
async def test_detect_source_language_returns_none_on_connection_error():
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(side_effect=httpx.ConnectError("boom"))
        async with httpx.AsyncClient() as client:
            lang = await detect_source_language(client, "https://example.com/page")
    assert lang is None


@pytest.mark.asyncio
async def test_detect_source_language_skips_non_text_content_type():
    with respx.mock() as mock:
        mock.get("https://example.com/file.pdf").mock(
            return_value=httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4")
        )
        async with httpx.AsyncClient() as client:
            lang = await detect_source_language(client, "https://example.com/file.pdf")
    assert lang is None


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_via_url():
    text = "{{Cite web|title=Some Page|url=https://example.com/page}}"
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-language": "en"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client)
    assert "|language=en" in result.patched_wikitext
    assert result.filled == {"Some Page": "en"}
    assert result.fetched_urls == 1


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_falls_back_to_title_when_no_url():
    text = "{{Cite book|title=This is quite clearly written in English prose}}"
    async with httpx.AsyncClient() as client:
        result = await fill_missing_citation_languages(text, client)
    assert "|language=en" in result.patched_wikitext
    assert result.fetched_urls == 0


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_skips_already_tagged():
    text = "{{Cite web|title=X|url=https://example.com/page|language=fr}}"
    async with httpx.AsyncClient() as client:
        result = await fill_missing_citation_languages(text, client)
    assert result.attempted == 0
    assert result.filled == {}
    assert result.patched_wikitext == text


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_ignores_non_citation_templates():
    text = "{{Infobox royalty|name=X}}"
    async with httpx.AsyncClient() as client:
        result = await fill_missing_citation_languages(text, client)
    assert result.attempted == 0
    assert result.patched_wikitext == text


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_respects_max_url_fetches():
    text = "".join(
        f"{{{{Cite web|title=Page {i}|url=https://example.com/{i}}}}}" for i in range(5)
    )
    with respx.mock() as mock:
        mock.get(url__regex=r"https://example\.com/\d+").mock(
            return_value=httpx.Response(200, headers={"content-language": "en"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client, max_url_fetches=2)
    assert result.fetched_urls == 2
    # the other 3 should still get a fallback attempt from their (too-short) titles, likely unfilled
    assert result.attempted == 5


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_failed_fetch_falls_back_to_title():
    text = "{{Cite web|title=This is clearly an English sentence for testing|url=https://example.com/dead}}"
    with respx.mock() as mock:
        mock.get("https://example.com/dead").mock(side_effect=httpx.ConnectError("boom"))
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client)
    assert result.filled == {"This is clearly an English sentence for testing": "en"}


@pytest.mark.asyncio
async def test_title_wins_over_url_when_they_disagree():
    # Regression: a Springer/JSTOR/Cambridge-style landing page reports its
    # own English UI language via <html lang>, not the cited work's actual
    # language — the title text is direct evidence about the work itself.
    german_title = "Sonderheft zum 60. Geburtstag von Herrn Professor mit vielen Wörtern"
    text = f"{{{{Cite book|title={german_title}|url=https://example.com/landing}}}}"
    with respx.mock() as mock:
        mock.get("https://example.com/landing").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/html"}, text='<html lang="en"><body>x</body></html>'
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client)
    assert result.filled == {german_title: "de"}
    assert "|language=de" in result.patched_wikitext


@pytest.mark.asyncio
async def test_url_used_when_title_too_short_to_guess():
    text = "{{Cite web|title=Home|url=https://example.com/page}}"
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-language": "sr"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client)
    assert result.filled == {"Home": "sr"}


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_preserves_other_params():
    text = "{{Cite web|title=Some Page|url=https://example.com/page|access-date=2024-01-01}}"
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-language": "en"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client)
    assert "access-date=2024-01-01" in result.patched_wikitext
    assert "title=Some Page" in result.patched_wikitext


def test_fix_citation_param_names_real_world_case():
    # Exact case observed in a real run: every CS1 param name mistranslated.
    text = (
        "{{Cite web |data=3 mars 2025 |titulli=Some Title "
        "|url=https://example.com |faqja=BalkanWeb "
        "|data-e-përdorimit=22 qershor 2026|language=en }}"
    )
    result = fix_citation_param_names(text)
    assert "|date=3 mars 2025" in result.patched_wikitext
    assert "|title=Some Title" in result.patched_wikitext
    assert "|website=BalkanWeb" in result.patched_wikitext
    assert "|access-date=22 qershor 2026" in result.patched_wikitext
    assert result.renamed == {
        "data": "date",
        "titulli": "title",
        "faqja": "website",
        "data-e-përdorimit": "access-date",
    }


def test_fix_citation_param_names_preserves_values_and_order():
    text = "{{Cite book|titulli=Book Name|botues=Some Press|vit=2020}}"
    result = fix_citation_param_names(text)
    assert result.patched_wikitext == "{{Cite book|title=Book Name|publisher=Some Press|year=2020}}"


def test_fix_citation_param_names_no_changes_when_already_english():
    text = "{{Cite web|title=X|url=https://example.com|access-date=2024-01-01}}"
    result = fix_citation_param_names(text)
    assert result.patched_wikitext == text
    assert result.renamed == {}


def test_fix_citation_param_names_skips_rename_if_english_already_present():
    # Both |title= and |titulli= present — ambiguous, not the harness's
    # call to resolve which one is authoritative, so leave both alone.
    text = "{{Cite web|title=Correct|titulli=Duplicate}}"
    result = fix_citation_param_names(text)
    assert result.patched_wikitext == text
    assert result.renamed == {}


def test_fix_citation_param_names_ignores_non_citation_templates():
    text = "{{Infobox person|titulli=Not a citation param}}"
    result = fix_citation_param_names(text)
    assert result.patched_wikitext == text
    assert result.renamed == {}


def test_fix_citation_param_names_is_case_insensitive_on_param_name():
    text = "{{Cite web|Titulli=Some Title}}"
    result = fix_citation_param_names(text)
    assert "title=Some Title" in result.patched_wikitext


def test_fix_citation_param_names_multiple_citations():
    text = "{{Cite web|titulli=A}} and {{Cite book|botues=B}}"
    result = fix_citation_param_names(text)
    assert "title=A" in result.patched_wikitext
    assert "publisher=B" in result.patched_wikitext
    assert result.renamed == {"titulli": "title", "botues": "publisher"}


def test_dedupe_short_footnotes_real_world_collision():
    text = (
        '{{sfn|Lane Fox|2011|p=342|ps=: "Quote version A"}}'
        " text in between "
        '{{sfn|Lane Fox|2011|p=342|ps=: "Quote version B, different"}}'
    )
    result = dedupe_short_footnotes(text)
    assert result.patched_wikitext.count('ps=: "Quote version A"') == 2
    assert "Quote version B" not in result.patched_wikitext
    assert len(result.canonicalized) == 1


def test_dedupe_short_footnotes_no_change_when_already_consistent():
    text = '{{sfn|Smith|2020|p=5|ps=same}} and {{sfn|Smith|2020|p=5|ps=same}}'
    result = dedupe_short_footnotes(text)
    assert result.patched_wikitext == text
    assert result.canonicalized == {}


def test_dedupe_short_footnotes_different_citations_untouched():
    # different page -> different identity, not a collision at all
    text = '{{sfn|Smith|2020|p=5|ps=A}} and {{sfn|Smith|2020|p=6|ps=B}}'
    result = dedupe_short_footnotes(text)
    assert result.patched_wikitext == text
    assert result.canonicalized == {}


def test_dedupe_short_footnotes_handles_harvnb():
    text = (
        '{{harvnb|Jones|1999|p=10|ps=First version}} '
        '{{harvnb|Jones|1999|p=10|ps=Second version}}'
    )
    result = dedupe_short_footnotes(text)
    assert result.patched_wikitext.count("First version") == 2
    assert "Second version" not in result.patched_wikitext


def test_dedupe_short_footnotes_adds_ps_when_one_call_lacks_it():
    text = '{{sfn|Smith|2020|p=5}} and {{sfn|Smith|2020|p=5|ps=extra text}}'
    result = dedupe_short_footnotes(text)
    # first call (no ps=) is canonical -> ps= removed from the second too
    assert "ps=" not in result.patched_wikitext
    assert len(result.canonicalized) == 1


def test_dedupe_short_footnotes_three_or_more_calls():
    text = (
        '{{sfn|X|2000|p=1|ps=A}} {{sfn|X|2000|p=1|ps=B}} {{sfn|X|2000|p=1|ps=C}}'
    )
    result = dedupe_short_footnotes(text)
    assert result.patched_wikitext.count("ps=A") == 3
    assert "ps=B" not in result.patched_wikitext
    assert "ps=C" not in result.patched_wikitext
