import httpx
import pytest
import respx

from wiki_translate_harness.models import ModelPricing
from wiki_translate_harness.openrouter import OpenRouterClient, OpenRouterError, compute_cost

BASE_URL = "https://openrouter.ai/api/v1"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr("wiki_translate_harness.openrouter.asyncio.sleep", fast_sleep)


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


def test_compute_cost():
    pricing = ModelPricing(model_id="m", prompt_price_per_token=0.000001, completion_price_per_token=0.000002)
    cost = compute_cost(pricing, 1000, 500)
    assert cost == pytest.approx(0.000001 * 1000 + 0.000002 * 500)


def test_compute_cost_none_pricing():
    assert compute_cost(None, 1000, 500) == 0.0
