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


def _ok(text="përshëndetje") -> OpenCodeCLIResult:
    return OpenCodeCLIResult(is_error=False, result_text=text, model_used="auto")


def _err(stderr="boom") -> OpenCodeCLIResult:
    return OpenCodeCLIResult(is_error=True, model_used="auto", stderr=stderr)


@pytest.mark.asyncio
async def test_successful_call_returns_text_and_zero_usage(monkeypatch):
    monkeypatch.setattr(
        "wiki_translation_harness.opencode_go_client.run_opencode_cli",
        lambda *a, **kw: _ok(),
    )
    client = OpenCodeGoClient(model="auto")
    text, pt, ct = await client.chat_completion("auto", _MESSAGES)
    assert text == "përshëndetje"
    assert pt == 0
    assert ct == 0


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


def test_run_opencode_cli_omits_model_flag_for_auto_sentinel():
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
    ) as mock_run:
        run_opencode_cli("system", "user", model="auto")

    cmd = mock_run.call_args[0][0]
    assert "--model" not in cmd


def test_run_opencode_cli_passes_explicit_model():
    with patch(
        "wiki_translation_harness.opencode_go_client.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
    ) as mock_run:
        run_opencode_cli("system", "user", model="anthropic/claude-sonnet-4-5")

    cmd = mock_run.call_args[0][0]
    assert "--model" in cmd
    assert "anthropic/claude-sonnet-4-5" in cmd


def test_run_opencode_cli_nonzero_exit_is_error():
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
