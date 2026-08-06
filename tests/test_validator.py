from wiki_translate_harness.validator import format_errors, validate_wikitext


def test_valid_wikitext_passes():
    text = "'''Bold''' [[Link|Display]] {{cite web|title=x}} <ref>foo</ref> {| |-\n| a\n|} <!-- c -->"
    result = validate_wikitext(text)
    assert result.valid
    assert result.issues == []


def test_unbalanced_link_detected():
    result = validate_wikitext("[[Broken link")
    assert not result.valid
    assert any(i.kind == "link" for i in result.issues)


def test_unbalanced_template_detected():
    result = validate_wikitext("{{cite web|title=x")
    assert not result.valid
    assert any(i.kind == "template" for i in result.issues)


def test_unbalanced_table_detected():
    result = validate_wikitext("{| class=wikitable\n|-\n| a\n")
    assert not result.valid
    assert any(i.kind == "table" for i in result.issues)


def test_unbalanced_ref_detected():
    result = validate_wikitext("text <ref>unterminated")
    assert not result.valid
    assert any(i.kind == "reference" for i in result.issues)


def test_self_closing_ref_is_balanced():
    result = validate_wikitext('text <ref name="x" /> more')
    assert result.valid


def test_unbalanced_comment_detected():
    result = validate_wikitext("<!-- unterminated comment")
    assert not result.valid
    assert any(i.kind == "comment" for i in result.issues)


def test_format_errors_readable():
    result = validate_wikitext("[[broken")
    errs = format_errors(result)
    assert len(errs) >= 1
    assert "link" in errs[0]


def test_heading_merged_onto_prior_line_detected():
    # Regression: a chunk boundary that drops the newline before a heading
    # (e.g. trailing content ending in "-->" immediately followed by "==...==")
    # silently fails to render as a heading in real MediaWiki.
    result = validate_wikitext("some trailing text-->== Referime ==\n{{reflist}}\n")
    assert not result.valid
    assert any(i.kind == "heading" for i in result.issues)


def test_heading_at_true_line_start_not_flagged():
    result = validate_wikitext("Lead text.\n\n== History ==\nBody.\n\n=== Sub ===\nMore.\n")
    assert result.valid


def test_heading_at_start_of_document_not_flagged():
    result = validate_wikitext("== Lead ==\nBody text.\n")
    assert result.valid


def test_equals_in_template_param_not_flagged_as_heading():
    result = validate_wikitext("Text with {{infobox|param=value}} more text.\n")
    assert result.valid


def test_leaked_attribution_block_detected():
    text = (
        "Some translated prose.\n\n"
        "<!--\nNUK ËSHTË PJESË E ARTIKULLIT — mos e kopjo këtë bllok në faqen kryesore.\n\n"
        "== Për faqen e diskutimit (Talk) ==\n"
        "{{Përkthyer nga|en|Test|1 janar 2024|123}}\n-->"
    )
    result = validate_wikitext(text)
    assert not result.valid
    assert any(i.kind == "leaked_commentary" for i in result.issues)


def test_leaked_fabrication_refusal_detected():
    text = "Nuk ka tekst burimor për të përkthyer në këtë seksion."
    result = validate_wikitext(text)
    assert not result.valid
    assert any(i.kind == "leaked_commentary" for i in result.issues)


def test_leaked_english_refusal_detected():
    text = "I cannot translate this section as there is no source text provided."
    result = validate_wikitext(text)
    assert not result.valid


def test_normal_prose_not_flagged_as_leaked_commentary():
    text = "'''Bardhyli''' ishte një mbret ilir i shekullit IV p.e.s.\n\n== Familja ==\nAi pati disa fëmijë."
    result = validate_wikitext(text)
    assert result.valid


def test_degenerate_repetition_loop_detected():
    # Regression: a real chunk (Timeline of Albanian history, "20th
    # century") came back from the model as thousands of near-identical
    # repeated lines instead of a translation — no fixed phrase to match,
    # since the repeated content itself varies run to run, so this needs a
    # content-agnostic detector rather than another string in the marker
    # list. zlib compression ratio catches it: real wikitext (even
    # formatting-heavy tables) compresses to ~30-47% of raw size, while
    # this kind of pathological loop compresses to under 3%.
    text = ("- cite a book in the context (Albanopolis in the list of links in this section) " * 400)
    result = validate_wikitext(text)
    assert not result.valid
    assert any(i.kind == "degenerate_repetition" for i in result.issues)


def test_normal_long_prose_not_flagged_as_degenerate_repetition():
    # Varied content, including a formatting-heavy table (structurally
    # repetitive but with different data per row, like real wikitext) —
    # must stay well clear of the compression-ratio threshold.
    rows = "\n".join(f"| {2000 + i} || Ngjarje e rëndësishme numër {i} në histori || fq. {i * 3}" for i in range(80))
    text = (
        "'''Bardhyli''' ishte një mbret ilir i shekullit IV p.e.s. Ai qeverisi mbretërinë Dardane "
        "dhe u përfshi në luftëra kundër Maqedonisë antike nën udhëheqjen e Filipit II.\n\n"
        "== Ngjarjet ==\n{| class=\"wikitable\"\n" + rows + "\n|}\n"
    )
    result = validate_wikitext(text)
    assert result.valid
