import pytest

from wiki_translation_harness.review import build_review_messages, parse_review_findings, review_article


class FakeReviewClient:
    """Matches test_translator.py's fake — run_completion calls
    client.chat_completion(model, messages, temperature, on_retry=...)."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_completion(self, model, messages, temperature=0.0, on_retry=None, usage_out=None):
        self.calls.append(messages)
        text = self.responses.pop(0)
        return text, 100, 50


def test_build_review_messages_includes_confirmed_template_params():
    messages = build_review_messages(
        "src", "translated", "Article", "sq", {"Infobox settlement": ["name", "population"]}
    )
    user = messages[1]["content"]
    assert "Infobox settlement" in user
    assert "population" in user


def test_parse_review_findings_from_fenced_json():
    text = (
        "Here are the findings:\n\n```json\n"
        '[{"kind": "semantic_fidelity", "severity": "error", '
        '"message": "Dropped a fact.", "snippet": "diçka"}]\n```'
    )
    issues = parse_review_findings(text)
    assert len(issues) == 1
    assert issues[0].kind == "semantic_fidelity"
    assert issues[0].severity == "error"
    assert issues[0].snippet == "diçka"


def test_parse_review_findings_invalid_json_returns_empty():
    assert parse_review_findings("not json at all") == []


def test_parse_review_findings_unknown_kind_falls_back_to_semantic_fidelity():
    text = '```json\n[{"kind": "bogus_kind", "message": "x"}]\n```'
    issues = parse_review_findings(text)
    assert issues[0].kind == "semantic_fidelity"


@pytest.mark.asyncio
async def test_review_article_returns_findings_and_result():
    client = FakeReviewClient(['```json\n[{"kind": "grammar_case", "message": "Wrong case."}]\n```'])
    findings, result = await review_article(
        client, "claude-sonnet-5", 0.0, "source", "translated", "Article", "sq", {}, None
    )
    assert len(findings) == 1
    assert findings[0].message == "Wrong case."
    assert result.model == "claude-sonnet-5"
    assert result.prompt_tokens == 100
