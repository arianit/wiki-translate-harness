"""Claude Code CLI engine: runs `claude -p` non-interactively under the
caller's existing Claude Code subscription/login. No API key used or
required.

The subprocess-invocation logic (temp-file system prompt, stdin user
prompt, stream-json parsing, the truncation heuristic) is ported from
multimodeltranslationpipeline's mmtp/claude_cli.py — a proven pattern
already running in production for a sibling translation pipeline. Ported
rather than imported: these are independently-deployable projects, not a
shared library.

Exposes ClaudeCodeClient, satisfying the same duck-typed contract
OpenRouterClient does (see engines.LLMEngineClient) so translator.py's
translate_chunk()/repair.py's repair_chunk() need no changes at all to use
either engine.
"""
from __future__ import annotations

import asyncio
import json
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wiki_translate_harness.models import EngineError, ModelPricing
from wiki_translate_harness.openrouter import RetryCallback

# See mmtp/claude_cli.py's module docstring for the full incident writeup:
# plain `--output-format json`'s `result` field only reflects the LAST
# internal turn a generation needed, silently dropping every earlier turn's
# text even though it was billed. --output-format stream-json (+ --verbose,
# mandatory alongside it in print mode) instead emits one event per content
# block as it's produced; concatenating every `assistant` event's text
# blocks, in order, reconstructs the true complete output regardless of how
# many internal turns it took.
_MIN_CHARS_PER_OUTPUT_TOKEN = 2.0
_TRUNCATION_CHECK_MIN_OUTPUT_TOKENS = 2000


class ClaudeCodeError(EngineError):
    pass


@dataclass
class ClaudeCLIResult:
    is_error: bool
    result_text: str = ""
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    model_used: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    stderr: str = ""


def run_claude_cli(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    cli_path: str = "claude",
    permission_mode: str = "bypassPermissions",
    timeout_s: float = 600.0,
) -> ClaudeCLIResult:
    """Single-shot, tool-less, blocking call — see ClaudeCodeClient.chat_completion
    for why this runs off the event loop via asyncio.to_thread rather than
    being called directly from async code.

    --tools "" disables all tools (no accidental file writes, no live
    fetching, no tool-round-trip latency — the chunk's wikitext is already
    embedded in user_prompt, there is nothing to fetch live). System prompt
    goes to a temp file (--system-prompt-file) and the user prompt goes over
    stdin, since full articles plus skill content can exceed the OS argv
    size limit."""
    cmd = [
        cli_path,
        "-p",
        "--model",
        model,
        "--tools",
        "",
        "--permission-mode",
        permission_mode,
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]

    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", prefix="wth-system-prompt-", delete=False
        ) as f:
            f.write(system_prompt)
            system_prompt_path = f.name
        try:
            proc = subprocess.run(
                [*cmd, "--system-prompt-file", system_prompt_path],
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        finally:
            Path(system_prompt_path).unlink(missing_ok=True)
    except subprocess.TimeoutExpired as exc:
        return ClaudeCLIResult(
            is_error=True,
            model_used=model,
            stderr=f"claude CLI call timed out after {timeout_s}s: {exc}",
        )
    except FileNotFoundError as exc:
        return ClaudeCLIResult(
            is_error=True,
            model_used=model,
            stderr=f"claude CLI not found ({cli_path!r}): {exc}",
        )

    text_blocks: list[str] = []
    result_event: dict | None = None
    saw_any_event = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        saw_any_event = True
        event_type = event.get("type")
        if event_type == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and block.get("text"):
                    text_blocks.append(block["text"])
        elif event_type == "result":
            result_event = event

    if not saw_any_event or result_event is None:
        return ClaudeCLIResult(
            is_error=True,
            model_used=model,
            stderr=(proc.stderr or proc.stdout or "no output")[:2000],
        )

    data = result_event
    reassembled_text = "".join(text_blocks)
    # Fall back to the (possibly turn-truncated) top-level `result` field
    # only if no streamed text blocks were captured at all.
    result_text = reassembled_text or (data.get("result") or "")

    usage = data.get("usage") or {}
    output_tokens = usage.get("output_tokens", 0)
    # Prompt caching means most of a call's real input (here: the ~70KB
    # skill text in the system prompt) lands in cache_creation_input_tokens
    # (first time) or cache_read_input_tokens (cache hit on a later call),
    # not input_tokens -- confirmed live: input_tokens alone reported as 2
    # while cache_creation_input_tokens was 30278 for the same call. Only
    # counting input_tokens made the caller's output/input ratio safety
    # check (translation-harness's max_token_ratio) see ~4882:1 and reject
    # a completely normal call as a suspected runaway generation. All three
    # figures reflect tokens that were genuinely part of the input context,
    # cached or not, so all three belong in what gets reported back.
    input_tokens = (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )
    cost = data.get("total_cost_usd") or 0.0
    duration_ms = data.get("duration_ms", 0)

    # Extended-thinking tokens (usage.output_tokens_details.thinking_tokens)
    # count toward output_tokens but produce zero characters in the text
    # stream -- confirmed live: a genuinely complete, well-formed 9515
    # output-token response (is_error: false, stop_reason: end_turn) was
    # 7765 of those tokens thinking, leaving ~1750 tokens' worth of real
    # text -- comparing chars-recovered against the full output_tokens
    # figure (mmtp's original heuristic, ported from a pipeline that
    # apparently doesn't see this much extended thinking) flagged it as
    # truncated even though nothing was lost. Only the non-thinking tokens
    # could ever have produced visible text, so only those belong in the
    # ratio check.
    thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0)
    visible_output_tokens = max(output_tokens - thinking_tokens, 0)

    if (
        visible_output_tokens >= _TRUNCATION_CHECK_MIN_OUTPUT_TOKENS
        and len(result_text) < visible_output_tokens * _MIN_CHARS_PER_OUTPUT_TOKEN
    ):
        return ClaudeCLIResult(
            is_error=True,
            total_cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            model_used=model,
            raw=data,
            stderr=(
                f"Suspected truncated output: {visible_output_tokens} non-thinking output "
                f"tokens ({output_tokens} total, {thinking_tokens} thinking) but only "
                f"{len(result_text)} chars of text were recovered from the stream — "
                "some content was likely lost."
            ),
        )

    cli_reported_error = bool(data.get("is_error"))
    return ClaudeCLIResult(
        is_error=cli_reported_error,
        result_text=result_text,
        total_cost_usd=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        model_used=model,
        raw=data,
        stderr=proc.stderr or "",
    )


class ClaudeCodeClient:
    def __init__(
        self,
        model: str,
        cli_path: str = "claude",
        permission_mode: str = "bypassPermissions",
        timeout_s: float = 600.0,
        max_retries: int = 5,
    ):
        self.model = model
        self.cli_path = cli_path
        self.permission_mode = permission_mode
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        on_retry: RetryCallback | None = None,
    ) -> tuple[str, int, int]:
        """Returns (text, prompt_tokens, completion_tokens). temperature is
        ignored — the CLI has no equivalent sampling-temperature flag.

        messages is always exactly [{"role":"system",...},{"role":"user",...}]
        — the fixed shape skill_loader.build_translation_messages/
        build_repair_messages produce."""
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
        if user_prompt is None:
            raise ClaudeCodeError(f"chat_completion got no user-role message: {messages!r}")

        attempt = 0
        while True:
            # subprocess.run() blocks; run it off the event loop so
            # concurrent chunk translations (translate_chunk's slot-queue
            # workers) don't get serialized behind one call.
            result = await asyncio.to_thread(
                run_claude_cli,
                system_prompt,
                user_prompt,
                model=model,
                cli_path=self.cli_path,
                permission_mode=self.permission_mode,
                timeout_s=self.timeout_s,
            )
            if not result.is_error:
                return result.result_text, result.input_tokens, result.output_tokens

            attempt += 1
            if "claude CLI not found" in result.stderr:
                # A missing binary won't fix itself on retry.
                raise ClaudeCodeError(result.stderr)
            if attempt > self.max_retries:
                raise ClaudeCodeError(
                    f"claude CLI call failed after {attempt} attempts: {result.stderr}"
                )
            await self._backoff(attempt, on_retry, reason=result.stderr or "unknown error")

    async def _backoff(self, attempt: int, on_retry: RetryCallback | None, reason: str) -> None:
        delay = min(2 ** (attempt - 1), 60) + random.uniform(0, 1)
        if on_retry is not None:
            on_retry(attempt, reason, delay)
        await asyncio.sleep(delay)

    async def get_pricing_for(self, model: str) -> ModelPricing | None:
        # No pricing table — the CLI's own result carries a real
        # total_cost_usd per call, but run_completion()'s cost model
        # (openrouter.compute_cost) multiplies an external per-token price
        # table instead of reading a provider-reported total, so cost
        # reports as $0 here, same as the existing "local" provider already
        # does. A future improvement could thread the CLI's own per-call
        # cost through TranslationResult instead of relying on this.
        return None

    async def fetch_pricing(self) -> dict[str, ModelPricing]:
        return {}

    async def aclose(self) -> None:
        pass  # no persistent connection to close
