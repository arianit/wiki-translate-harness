import httpx
import pytest
import respx

from wiki_translation_harness.models import InsufficientCreditsError, ModelPricing
from wiki_translation_harness.openrouter import OpenRouterClient, OpenRouterError, compute_cost, run_completion

BASE_URL = "https://openrouter.ai/api/v1"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr("wiki_translation_harness.openrouter.asyncio.sleep", fast_sleep)


def _success_response():
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "translated text"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


@pytest.mark.asyncio
async def test_successful_call_returns_usage():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(return_value=_success_response())
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=3)
        text, pt, ct = await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert text == "translated text"
        assert pt == 100
        assert ct == 50
        await client.aclose()


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds():
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(
            side_effect=[httpx.Response(429, text="rate limited"), _success_response()]
        )
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=3)
        text, _, _ = await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert text == "translated text"
        assert route.call_count == 2
        await client.aclose()


@pytest.mark.asyncio
async def test_retries_on_502_and_503():
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(502, text="bad gateway"),
                httpx.Response(503, text="unavailable"),
                _success_response(),
            ]
        )
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=5)
        text, _, _ = await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert text == "translated text"
        assert route.call_count == 3
        await client.aclose()


@pytest.mark.asyncio
async def test_gives_up_after_max_retries():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(return_value=httpx.Response(500, text="server error"))
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=2)
        with pytest.raises(OpenRouterError):
            await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        await client.aclose()


@pytest.mark.asyncio
async def test_retries_on_connection_error():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[httpx.ConnectError("boom"), _success_response()]
        )
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=3)
        text, _, _ = await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert text == "translated text"
        await client.aclose()


@pytest.mark.asyncio
async def test_non_retryable_4xx_raises_immediately():
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(return_value=httpx.Response(401, text="unauthorized"))
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=3)
        with pytest.raises(OpenRouterError):
            await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert route.call_count == 1
        await client.aclose()


@pytest.mark.asyncio
async def test_402_raises_insufficient_credits_not_generic_engine_error():
    # A distinct exception type, not just any OpenRouterError: pipeline.py
    # catches this one specifically to offer a fallback-provider switch
    # instead of just failing the chunk like any other engine error.
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(402, json={"error": {"message": "requires more credits"}})
        )
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=3)
        with pytest.raises(InsufficientCreditsError):
            await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert route.call_count == 1  # not retried -- retrying the same account can't help
        await client.aclose()


@pytest.mark.asyncio
async def test_429_with_insufficient_quota_code_raises_insufficient_credits():
    # Experiential Labs signals "out of credits" on HTTP 429 with
    # error.code == "insufficient_quota" -- unlike OpenRouter's plain 402,
    # 429 is otherwise a retryable status, so this must be checked before
    # the generic RETRYABLE_STATUS_CODES branch retries it away.
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "insufficient_credits: add funds", "code": "insufficient_quota"}},
            )
        )
        client = OpenRouterClient("xpl_test", BASE_URL, "test-agent/1.0", max_retries=3, provider="experiential")
        with pytest.raises(InsufficientCreditsError):
            await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert route.call_count == 1  # not retried
        await client.aclose()


@pytest.mark.asyncio
async def test_429_without_insufficient_quota_code_is_retried():
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json={"error": {"message": "slow down", "code": "org_rate_limit"}}),
                _success_response(),
            ]
        )
        client = OpenRouterClient("xpl_test", BASE_URL, "test-agent/1.0", max_retries=3, provider="experiential")
        text, _, _ = await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert text == "translated text"
        assert route.call_count == 2
        await client.aclose()


@pytest.mark.asyncio
async def test_experiential_request_carries_safety_identifier():
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(return_value=_success_response())
        client = OpenRouterClient("xpl_test", BASE_URL, "test-agent/1.0", max_retries=3, provider="experiential")
        await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        assert route.calls.last.request.content
        import json as _json

        body = _json.loads(route.calls.last.request.content)
        assert body["safety_identifier"] == "wiki-translation-harness"
        await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_request_has_no_safety_identifier():
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.post("/chat/completions").mock(return_value=_success_response())
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=3)
        await client.chat_completion("m", [{"role": "user", "content": "hi"}])
        import json as _json

        body = _json.loads(route.calls.last.request.content)
        assert "safety_identifier" not in body
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_fills_usage_out_with_raw_usage():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "translated text"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0042},
                },
            )
        )
        client = OpenRouterClient("xpl_test", BASE_URL, "test-agent/1.0", max_retries=3, provider="experiential")
        usage: dict = {}
        await client.chat_completion("m", [{"role": "user", "content": "hi"}], usage_out=usage)
        assert usage["cost"] == pytest.approx(0.0042)
        await client.aclose()


@pytest.mark.asyncio
async def test_run_completion_prefers_reported_cost_over_pricing_table():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "translated text"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0042},
                },
            )
        )
        client = OpenRouterClient("xpl_test", BASE_URL, "test-agent/1.0", max_retries=3, provider="experiential")
        # A pricing table is still passed in (as build_config's cost-estimate
        # step would), but the inline usage.cost must win.
        pricing = ModelPricing(model_id="m", prompt_price_per_token=1.0, completion_price_per_token=1.0)
        result = await run_completion(client, "m", [{"role": "user", "content": "hi"}], 0.0, pricing)
        assert result.cost_usd == pytest.approx(0.0042)
        await client.aclose()


@pytest.mark.asyncio
async def test_run_completion_falls_back_to_pricing_table_when_no_reported_cost():
    with respx.mock(base_url=BASE_URL) as mock:
        mock.post("/chat/completions").mock(return_value=_success_response())
        client = OpenRouterClient("sk-test", BASE_URL, "test-agent/1.0", max_retries=3)
        pricing = ModelPricing(model_id="m", prompt_price_per_token=0.000001, completion_price_per_token=0.000002)
        result = await run_completion(client, "m", [{"role": "user", "content": "hi"}], 0.0, pricing)
        assert result.cost_usd == pytest.approx(0.000001 * 100 + 0.000002 * 50)
        await client.aclose()


def test_compute_cost():
    pricing = ModelPricing(model_id="m", prompt_price_per_token=0.000001, completion_price_per_token=0.000002)
    cost = compute_cost(pricing, 1000, 500)
    assert cost == pytest.approx(0.000001 * 1000 + 0.000002 * 500)


def test_compute_cost_none_pricing():
    assert compute_cost(None, 1000, 500) == 0.0
