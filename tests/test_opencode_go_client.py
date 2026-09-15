import json
import subprocess
from unittest.mock import patch

import pytest

from wiki_translation_harness.models import InsufficientCreditsError
from wiki_translation_harness.opencode_go_client import (
    OpenCodeCLIResult,
    OpenCodeGoClient,
    OpenCodeGoError,
    OpenCodeGoSessionLimitError,
    run_opencode_cli,
)

_MESSAGES = [
    {"role": "system", "content": "You are a translator."},
    {"role": "user", "content": "Translate: hello"},
]


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr("wiki_translation_harness.opencode_go_client.asyncio.sleep", fast_sleep)


def _ok(text="përshëndetje", prompt_tokens=10, completion_tokens=5, cost_usd=None) -> OpenCodeCLIResult:
    return OpenCodeCLIResult(
        is_error=False,
        result_text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        model_used="auto",
    )


def _err(stderr="boom") -> OpenCodeCLIResult:
    return OpenCodeCLIResult(is_error=True, model_used="auto", stderr=stderr)


def _cli_events(*events: dict) -> subprocess.CompletedProcess:
    stdout = "\n".join(json.dumps(e) for e in events)
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


@pytest.mark.asyncio
async def test_successful_call_returns_text_and_usage(monkeypatch):
    monkeypatch.setattr(
        "wiki_translation_harness.opencode_go_client.run_opencode_cli",
        lambda *a, **kw: _ok(prompt_tokens=100, completion_tokens=42),
    )
    client = OpenCodeGoClient(model="auto")
    text, pt, ct = await client.chat_completion("auto", _MESSAGES)
    assert text == "përshëndetje"
    assert pt == 100
    assert ct == 42


@pytest.mark.asyncio
async def test_reported_cost_fills_usage_out(monkeypatch):
    monkeypatch.setattr(
        "wiki_translation_harness.opencode_go_client.run_opencode_cli",
        lambda *a, **kw: _ok(cost_usd=0.0042),
    )
    client = OpenCodeGoClient(model="auto")
    usage: dict = {}
    await client.chat_completion("auto", _MESSAGES, usage_out=usage)
    assert usage["cost"] == 0.0042


@pytest.mark.asyncio
async def test_no_reported_cost_leaves_usage_out_empty(monkeypatch):
    monkeypatch.setattr(
        "wiki_translation_harness.opencode_go_client.run_opencode_cli",
        lambda *a, **kw: _ok(cost_usd=None),
    )
    client = OpenCodeGoClient(model="auto")
    usage: dict = {}
    await client.chat_completion("auto", _MESSAGES, usage_out=usage)
    assert "cost" not in usage


@pytest.mark.asyncio
async def test_no_user_message_raises():
    client = OpenCodeGoClient(model="auto")
    with pytest.raises(OpenCodeGoError):
        await client.chat_completion("auto", [{"role": "system", "content": "x"}])


@pytest.mark.asyncio
async def test_missing_binary_fails_fast_without_retrying(monkeypatch):
    calls = []

    def fake(*a, **kw):
        calls.append(1)
        return _err(stderr="opencode CLI not found ('opencode'): [Errno 2] No such file or directory")

    monkeypatch.setattr("wiki_translation_harness.opencode_go_client.run_opencode_cli", fake)
    client = OpenCodeGoClient(model="auto", max_retries=5)
    with pytest.raises(OpenCodeGoError):
        await client.chat_completion("auto", _MESSAGES)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds(monkeypatch):
    results = [_err(), _err(), _ok()]

    def fake(*a, **kw):
        return results.pop(0)

    monkeypatch.setattr("wiki_translation_harness.opencode_go_client.run_opencode_cli", fake)
    retries_seen = []
    client = OpenCodeGoClient(model="auto", max_retries=5)
    text, pt, ct = await client.chat_completion(
        "auto", _MESSAGES, on_retry=lambda attempt, reason, delay: retries_seen.append(attempt)
    )
    assert text == "përshëndetje"
    assert retries_seen == [1, 2]


@pytest.mark.asyncio
async def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(
        "wiki_translation_harness.opencode_go_client.run_opencode_cli",
        lambda *a, **kw: _err(stderr="persistent failure"),
    )
    client = OpenCodeGoClient(model="auto", max_retries=2)
    with pytest.raises(OpenCodeGoError, match="persistent failure"):
        await client.chat_completion("auto", _MESSAGES)


@pytest.mark.asyncio
async def test_session_limit_fails_fast_without_retrying(monkeypatch):
    """A rate/usage-limit rejection cannot be fixed by backing off within
    the same run -- must raise immediately, and as an InsufficientCreditsError
    subtype so pipeline.py's ensure_fallback_engine() (if it were ever
    layered on top of an already-fallen-back-to opencode_go) would react."""
    calls = []

    def fake(*a, **kw):
        calls.append(1)
        return _err(stderr="Error: rate limit exceeded, please try again later")

    monkeypatch.setattr("wiki_translation_harness.opencode_go_client.run_opencode_cli", fake)
    client = OpenCodeGoClient(model="auto", max_retries=5)
    with pytest.raises(OpenCodeGoSessionLimitError):
        await client.chat_completion("auto", _MESSAGES)
    assert len(calls) == 1
    assert issubclass(OpenCodeGoSessionLimitError, InsufficientCreditsError)


@pytest.mark.asyncio
async def test_agent_flag_passed_through_to_cli(monkeypatch):
    seen_kwargs = {}

    def fake(*a, **kw):
        seen_kwargs.update(kw)
        return _ok()

    monkeypatch.setattr("wiki_translation_harness.opencode_go_client.run_opencode_cli", fake)
    client = OpenCodeGoClient(model="auto", agent="wiki-translation-harness")
    await client.chat_completion("auto", _MESSAGES)
    assert seen_kwargs["agent"] == "wiki-translation-harness"


def test_run_opencode_cli_omits_model_flag_for_auto_sentinel():
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=_cli_events(
            {"type": "step_start", "part": {}},
            {"type": "text", "part": {"text": "ok"}},
            {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
        ),
    ) as mock_run:
        run_opencode_cli("system", "user", model="auto")

    cmd = mock_run.call_args[0][0]
    assert "--model" not in cmd
    assert "--format" in cmd and "json" in cmd


def test_run_opencode_cli_passes_explicit_model_and_agent():
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=_cli_events(
            {"type": "text", "part": {"text": "ok"}},
            {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
        ),
    ) as mock_run:
        run_opencode_cli(
            "system", "user", model="anthropic/claude-sonnet-4-5", agent="wiki-translation-harness"
        )

    cmd = mock_run.call_args[0][0]
    assert "--model" in cmd
    assert "anthropic/claude-sonnet-4-5" in cmd
    assert "--agent" in cmd
    assert "wiki-translation-harness" in cmd


def test_run_opencode_cli_parses_real_event_shape():
    """Modeled directly on a real, live-captured `opencode run --format
    json` transcript (opencode v1.18.31): step_start -> text -> step_finish,
    with tokens.cache.{read,write} and a per-call cost."""
    events = [
        {
            "type": "step_start",
            "part": {"type": "step-start"},
        },
        {
            "type": "text",
            "part": {"type": "text", "text": "Hello!"},
        },
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "reason": "stop",
                "tokens": {
                    "total": 13208,
                    "input": 11307,
                    "output": 109,
                    "reasoning": 0,
                    "cache": {"write": 0, "read": 1792},
                },
                "cost": 0.0031,
            },
        },
    ]
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=_cli_events(*events),
    ):
        result = run_opencode_cli("system", "user", model="opencode/mimo-v2.5-free")

    assert not result.is_error
    assert result.result_text == "Hello!"
    assert result.prompt_tokens == 11307 + 1792  # input + cache.read
    assert result.completion_tokens == 109
    assert result.cost_usd == 0.0031


def test_run_opencode_cli_concatenates_multiple_text_events():
    events = [
        {"type": "text", "part": {"text": "part one. "}},
        {"type": "text", "part": {"text": "part two."}},
        {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
    ]
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=_cli_events(*events),
    ):
        result = run_opencode_cli("system", "user", model="auto")

    assert result.result_text == "part one. part two."


def test_run_opencode_cli_error_event_on_stdout_is_error():
    """Confirmed live: a bad-model rejection's error event lands on stdout,
    not stderr, alongside a non-zero exit code -- the default (non-JSON)
    format is the one that uses stderr instead."""
    events = [
        {
            "type": "error",
            "error": {
                "name": "UnknownError",
                "data": {"message": "Unexpected server error. Check server logs for details."},
            },
        },
    ]
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout="\n".join(json.dumps(e) for e in events), stderr=""
        ),
    ):
        result = run_opencode_cli("system", "user", model="bogus/no-such-model")

    assert result.is_error
    assert "Unexpected server error" in result.stderr


def test_run_opencode_cli_nonzero_exit_no_json_falls_back_to_stderr():
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom"),
    ):
        result = run_opencode_cli("system", "user", model="auto")

    assert result.is_error
    assert "boom" in result.stderr


def test_run_opencode_cli_missing_binary_reported_as_error():
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        side_effect=FileNotFoundError("no such file"),
    ):
        result = run_opencode_cli("system", "user", model="auto", cli_path="nonexistent-opencode")

    assert result.is_error
    assert "not found" in result.stderr


def test_run_opencode_cli_empty_stdin_message_fails_fast():
    """Confirmed live: `opencode run` with no positional message and empty
    stdin exits 1 immediately with a clear error rather than hanging on a
    TTY prompt."""
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Error: You must provide a message or a command"
        ),
    ):
        result = run_opencode_cli("", "", model="auto")

    assert result.is_error
