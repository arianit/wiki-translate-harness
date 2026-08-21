import pytest

from wiki_translate_harness.claude_code_client import (
    ClaudeCLIResult,
    ClaudeCodeClient,
    ClaudeCodeError,
)

_MESSAGES = [
    {"role": "system", "content": "You are a translator."},
    {"role": "user", "content": "Translate: hello"},
]


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr("wiki_translate_harness.claude_code_client.asyncio.sleep", fast_sleep)


def _ok(text="përshëndetje", input_tokens=10, output_tokens=5) -> ClaudeCLIResult:
    return ClaudeCLIResult(
        is_error=False,
        result_text=text,
        total_cost_usd=0.001,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=500,
        model_used="claude-sonnet-5",
    )


def _err(stderr="boom") -> ClaudeCLIResult:
    return ClaudeCLIResult(is_error=True, model_used="claude-sonnet-5", stderr=stderr)


@pytest.mark.asyncio
async def test_successful_call_returns_usage(monkeypatch):
    monkeypatch.setattr(
        "wiki_translate_harness.claude_code_client.run_claude_cli",
        lambda *a, **kw: _ok(),
    )
    client = ClaudeCodeClient(model="claude-sonnet-5")
    text, pt, ct = await client.chat_completion("claude-sonnet-5", _MESSAGES)
    assert text == "përshëndetje"
    assert pt == 10
    assert ct == 5


@pytest.mark.asyncio
async def test_no_user_message_raises():
    client = ClaudeCodeClient(model="claude-sonnet-5")
    with pytest.raises(ClaudeCodeError):
        await client.chat_completion("claude-sonnet-5", [{"role": "system", "content": "x"}])


@pytest.mark.asyncio
async def test_missing_binary_fails_fast_without_retrying(monkeypatch):
    calls = []

    def fake(*a, **kw):
        calls.append(1)
        return _err(stderr="claude CLI not found ('claude'): [Errno 2] No such file or directory")

    monkeypatch.setattr("wiki_translate_harness.claude_code_client.run_claude_cli", fake)
    client = ClaudeCodeClient(model="claude-sonnet-5", max_retries=5)
    with pytest.raises(ClaudeCodeError):
        await client.chat_completion("claude-sonnet-5", _MESSAGES)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds(monkeypatch):
    results = [_err(), _err(), _ok()]

    def fake(*a, **kw):
        return results.pop(0)

    monkeypatch.setattr("wiki_translate_harness.claude_code_client.run_claude_cli", fake)
    retries_seen = []
    client = ClaudeCodeClient(model="claude-sonnet-5", max_retries=5)
    text, pt, ct = await client.chat_completion(
        "claude-sonnet-5", _MESSAGES, on_retry=lambda attempt, reason, delay: retries_seen.append(attempt)
    )
    assert text == "përshëndetje"
    assert retries_seen == [1, 2]


@pytest.mark.asyncio
async def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(
        "wiki_translate_harness.claude_code_client.run_claude_cli",
        lambda *a, **kw: _err(stderr="persistent failure"),
    )
    client = ClaudeCodeClient(model="claude-sonnet-5", max_retries=2)
    with pytest.raises(ClaudeCodeError, match="persistent failure"):
        await client.chat_completion("claude-sonnet-5", _MESSAGES)


def test_truncation_heuristic_flags_short_result_as_error():
    """Ported truncation heuristic (secondary safety net for stream-parsing
    content loss, not the primary stop_reason=max_tokens signal), exercised
    directly against run_claude_cli's parsing logic via a hand-built
    stream-json payload, since it's internal to that function rather than
    ClaudeCodeClient."""
    import json
    import subprocess
    from unittest.mock import patch

    from wiki_translate_harness.claude_code_client import run_claude_cli

    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "short"}]}},
        {"type": "result", "usage": {"input_tokens": 100, "output_tokens": 3000}, "total_cost_usd": 0.05},
    ]
    stdout = "\n".join(json.dumps(e) for e in events)

    with patch(
        "wiki_translate_harness.claude_code_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
    ):
        result = run_claude_cli("system", "user", model="claude-sonnet-5")

    assert result.is_error
    assert "content loss" in result.stderr.lower()


def test_max_tokens_stop_reason_flagged_as_truncated():
    """stop_reason=max_tokens is the authoritative truncation signal
    (straight from the API) -- must be flagged even when the reconstructed
    text is nowhere near short enough to trip the chars-per-token safety
    net on its own."""
    import json
    import subprocess
    from unittest.mock import patch

    from wiki_translate_harness.claude_code_client import run_claude_cli

    long_text = "x" * 20000
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": long_text}]}},
        {
            "type": "result",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 100, "output_tokens": 8000},
            "total_cost_usd": 0.2,
        },
    ]
    stdout = "\n".join(json.dumps(e) for e in events)

    with patch(
        "wiki_translate_harness.claude_code_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
    ):
        result = run_claude_cli("system", "user", model="claude-sonnet-5")

    assert result.is_error
    assert "max_tokens" in result.stderr


def test_end_turn_with_moderate_ratio_not_flagged():
    """Regression test for a real false positive: a genuinely complete,
    naturally-finished response (stop_reason: end_turn) at a ~1.3
    chars/token ratio -- below mmtp's original 2.0 threshold but above the
    lowered 1.0 -- confirmed live that Albanian wikitext output can
    legitimately run this dense (a real reproduction measured 1.94 for a
    confirmed-complete response) -- must not be flagged. Uses output_tokens
    above the 2000 noise floor so the ratio check actually runs."""
    import json
    import subprocess
    from unittest.mock import patch

    from wiki_translate_harness.claude_code_client import run_claude_cli

    text = "x" * 4000  # 4000 chars / 3000 visible tokens = 1.33 -- between 1.0 and 2.0
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {
            "type": "result",
            "is_error": False,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 3000},
            "total_cost_usd": 0.1,
        },
    ]
    stdout = "\n".join(json.dumps(e) for e in events)

    with patch(
        "wiki_translate_harness.claude_code_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
    ):
        result = run_claude_cli("system", "user", model="claude-sonnet-5")

    assert not result.is_error
    assert result.result_text == text


def test_thinking_tokens_excluded_from_truncation_check():
    """Regression test for a real false positive: a genuinely complete
    response (is_error: false, stop_reason: end_turn) with heavy extended
    thinking got flagged as truncated because thinking tokens count toward
    output_tokens but produce zero characters in the text stream. Modeled
    on a real reproduction: 9515 total output tokens, 7765 of them
    thinking, ~3500 chars of real text -- must NOT be flagged."""
    import json
    import subprocess
    from unittest.mock import patch

    from wiki_translate_harness.claude_code_client import run_claude_cli

    real_text = "x" * 3500  # ~3500 chars of "real" translated text
    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": real_text}]}},
        {
            "type": "result",
            "is_error": False,
            "usage": {
                "input_tokens": 2,
                "output_tokens": 9515,
                "output_tokens_details": {"thinking_tokens": 7765},
            },
            "total_cost_usd": 0.33,
        },
    ]
    stdout = "\n".join(json.dumps(e) for e in events)

    with patch(
        "wiki_translate_harness.claude_code_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
    ):
        result = run_claude_cli("system", "user", model="claude-sonnet-5")

    assert not result.is_error
    assert result.result_text == real_text


def test_cache_tokens_counted_as_input():
    """Regression test for a real false positive: prompt caching put most
    of a call's real input in cache_creation_input_tokens rather than
    input_tokens (confirmed live: input_tokens=2,
    cache_creation_input_tokens=30278 for the same call), which made
    translation-harness's output/input ratio safety check see ~4882:1 and
    reject a normal call as a suspected runaway generation."""
    import json
    import subprocess
    from unittest.mock import patch

    from wiki_translate_harness.claude_code_client import run_claude_cli

    events = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "translated text"}]}},
        {
            "type": "result",
            "is_error": False,
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 30278,
                "cache_read_input_tokens": 0,
                "output_tokens": 100,
            },
            "total_cost_usd": 0.05,
        },
    ]
    stdout = "\n".join(json.dumps(e) for e in events)

    with patch(
        "wiki_translate_harness.claude_code_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
    ):
        result = run_claude_cli("system", "user", model="claude-sonnet-5")

    assert result.input_tokens == 30280


def test_missing_cli_binary_reported_as_error():
    from unittest.mock import patch

    from wiki_translate_harness.claude_code_client import run_claude_cli

    with patch(
        "wiki_translate_harness.claude_code_client.subprocess.run",
        side_effect=FileNotFoundError("no such file"),
    ):
        result = run_claude_cli("system", "user", model="claude-sonnet-5", cli_path="nonexistent-claude")

    assert result.is_error
    assert "not found" in result.stderr
