"""OpenCode Go engine (github.com/sst/opencode's Go CLI): runs `opencode run`
non-interactively, under whichever model/provider the caller's own opencode
config already has authenticated.

This exists specifically as the harness's default fallback target when
`provider: claude_code` hits its own account-level session/rate limit (see
claude_code_client.py's ClaudeCodeSessionLimitError and pipeline.py's
ensure_fallback_engine) — opencode is a separate binary with its own,
independent auth/session, so it isn't affected by Claude Code hitting its
cap. It works standalone too via `--provider opencode_go`.

Unlike claude_code_client.py's `claude -p --output-format stream-json`, the
opencode CLI has no publicly-documented machine-readable single-shot output
format to build against here, so this deliberately treats it as a plain
Unix filter: the combined prompt goes over stdin, the full stdout is the
result text, and a non-zero exit code (or a recognizable rate/usage-limit
message on stderr) is the only error signal. That means no real token/cost
accounting is available for this engine (get_pricing_for returns None, same
as the "local" and claude_code engines) — a future revision can tighten this
against opencode's actual output contract if/when it grows one.

Exposes OpenCodeGoClient, satisfying the same duck-typed contract
OpenRouterClient/ClaudeCodeClient do (see engines.LLMEngineClient) so
translator.py's translate_chunk()/repair.py's repair_chunk() need no changes
to use it.
"""
from __future__ import annotations

import asyncio
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wiki_translation_harness.models import EngineError, InsufficientCreditsError, ModelPricing
from wiki_translation_harness.openrouter import RetryCallback

# Substrings looked for (case-insensitively) in a failed call's stderr/stdout
# to recognize an account/session-level usage limit -- as opposed to a
# transient failure worth retrying. Deliberately broad/lowercase-matched
# since, unlike Claude Code's structured `api_error_status` field, opencode
# gives no confirmed structured error shape to branch on here.
_SESSION_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "usage limit",
    "quota",
    "429",
    "spend limit",
)


class OpenCodeGoError(EngineError):
    pass


class OpenCodeGoSessionLimitError(OpenCodeGoError, InsufficientCreditsError):
    """opencode's own configured provider rejected the call for a
    rate/usage/spend-limit reason. Raised as an InsufficientCreditsError (not
    just a plain OpenCodeGoError) for interface parity with
    claude_code_client.ClaudeCodeSessionLimitError -- retrying the same
    engine can't succeed, so pipeline.py's ensure_fallback_engine() would
    otherwise be the right response, though since opencode_go is already
    used as the harness's fallback-of-last-resort, in practice this just
    fails the chunk fast instead of burning the retry budget."""


@dataclass
class OpenCodeCLIResult:
    is_error: bool
    result_text: str = ""
    duration_ms: int = 0
    model_used: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    stderr: str = ""


def run_opencode_cli(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    cli_path: str = "opencode",
    timeout_s: float = 600.0,
) -> OpenCodeCLIResult:
    """Single-shot, blocking call -- see OpenCodeGoClient.chat_completion for
    why this runs off the event loop via asyncio.to_thread rather than being
    called directly from async code.

    model == "auto" (config.default_model_for_provider's pick for this
    provider) omits --model entirely, letting `opencode` use its own
    configured default provider/model instead of a guessed id that opencode
    might not even be authenticated for.

    The system and user prompt are combined into a single stdin payload
    (opencode's `run` has no confirmed separate system-prompt flag the way
    Claude Code's CLI does) rather than passed as a positional argument,
    since full articles plus skill content can exceed the OS argv size
    limit -- same reasoning as claude_code_client.run_claude_cli."""
    cmd = [cli_path, "run"]
    if model != "auto":
        cmd += ["--model", model]

    combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

    try:
        proc = subprocess.run(
            cmd,
            input=combined_prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return OpenCodeCLIResult(
            is_error=True,
            model_used=model,
            stderr=f"opencode CLI call timed out after {timeout_s}s: {exc}",
        )
    except FileNotFoundError as exc:
        return OpenCodeCLIResult(
            is_error=True,
            model_used=model,
            stderr=f"opencode CLI not found ({cli_path!r}): {exc}",
        )

    if proc.returncode != 0:
        stderr_msg = (proc.stderr or proc.stdout or "unknown error").strip()[:2000]
        return OpenCodeCLIResult(
            is_error=True,
            model_used=model,
            stderr=stderr_msg or "opencode CLI exited non-zero with no output",
            raw={"returncode": proc.returncode},
        )

    return OpenCodeCLIResult(
        is_error=False,
        result_text=proc.stdout.strip(),
        model_used=model,
        stderr=proc.stderr or "",
    )


class OpenCodeGoClient:
    def __init__(
        self,
        model: str,
        cli_path: str = "opencode",
        timeout_s: float = 600.0,
        max_retries: int = 5,
        log_dir: Path | str | None = None,
    ):
        self.model = model
        self.cli_path = cli_path
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # Accepted for interface parity with ClaudeCodeClient (both are
        # constructed generically by engines.build_llm_client) but unused --
        # run_opencode_cli has no per-call diagnostic dump the way
        # claude_code_client._write_diagnostic_log does, since opencode's
        # stdout is already the plain result text, not a raw stream to
        # preserve for later inspection.
        self.log_dir = log_dir

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        on_retry: RetryCallback | None = None,
        usage_out: dict | None = None,
    ) -> tuple[str, int, int]:
        """Returns (text, prompt_tokens, completion_tokens). temperature is
        ignored (no CLI equivalent); prompt/completion token counts are
        always 0 -- see module docstring on the lack of a confirmed
        machine-readable output format to read real usage from. usage_out is
        accepted for interface parity with OpenRouterClient/ClaudeCodeClient
        but left unfilled.

        messages is always exactly [{"role":"system",...},{"role":"user",...}]
        -- the fixed shape skill_loader.build_translation_messages/
        build_repair_messages produce."""
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
        if user_prompt is None:
            raise OpenCodeGoError(f"chat_completion got no user-role message: {messages!r}")

        attempt = 0
        while True:
            # subprocess.run() blocks; run it off the event loop so
            # concurrent chunk translations (translate_chunk's slot-queue
            # workers) don't get serialized behind one call.
            result = await asyncio.to_thread(
                run_opencode_cli,
                system_prompt,
                user_prompt,
                model=model,
                cli_path=self.cli_path,
                timeout_s=self.timeout_s,
            )
            if not result.is_error:
                return result.result_text, 0, 0

            attempt += 1
            if "opencode CLI not found" in result.stderr:
                # A missing binary won't fix itself on retry.
                raise OpenCodeGoError(result.stderr)
            if any(marker in result.stderr.lower() for marker in _SESSION_LIMIT_MARKERS):
                # Same reasoning as claude_code_client's 429 handling: an
                # account/session-level limit cannot be fixed by backing off
                # within the same run, so fail fast rather than burn the
                # retry budget's wall time on a guaranteed-identical
                # rejection every time.
                raise OpenCodeGoSessionLimitError(
                    f"opencode session/usage limit: {result.stderr}"
                )
            if attempt > self.max_retries:
                raise OpenCodeGoError(
                    f"opencode CLI call failed after {attempt} attempts: {result.stderr}"
                )
            await self._backoff(attempt, on_retry, reason=result.stderr or "unknown error")

    async def _backoff(self, attempt: int, on_retry: RetryCallback | None, reason: str) -> None:
        delay = min(2 ** (attempt - 1), 60) + random.uniform(0, 1)
        if on_retry is not None:
            on_retry(attempt, reason, delay)
        await asyncio.sleep(delay)

    async def get_pricing_for(self, model: str) -> ModelPricing | None:
        # No pricing table -- opencode's own per-call cost, if any, isn't
        # exposed through a confirmed machine-readable output here. Same
        # $0-reported-cost tradeoff already accepted for "local" and
        # claude_code.
        return None

    async def fetch_pricing(self) -> dict[str, ModelPricing]:
        return {}

    async def aclose(self) -> None:
        pass  # no persistent connection to close
