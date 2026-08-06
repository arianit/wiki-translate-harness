import httpx
import pytest
import respx

from wiki_translate_harness.mediawiki import (
    MediaWikiClient,
    MediaWikiClientPool,
    MediaWikiError,
    wiki_api_url_for_lang,
)

API_URL = "https://en.wikipedia.org/w/api.php"


def test_wiki_api_url_for_lang():
    assert wiki_api_url_for_lang("sq") == "https://sq.wikipedia.org/w/api.php"
    assert wiki_api_url_for_lang("en") == "https://en.wikipedia.org/w/api.php"


def test_client_pool_reuses_client_for_same_lang():
    pool = MediaWikiClientPool("test-agent/1.0", default_lang="en")
    a = pool.get("sq")
    b = pool.get("sq")
    assert a is b


def test_client_pool_different_lang_different_client():
    pool = MediaWikiClientPool("test-agent/1.0", default_lang="en")
    en_client = pool.get("en")
    sq_client = pool.get("sq")
    assert en_client is not sq_client
    assert en_client.api_url == "https://en.wikipedia.org/w/api.php"
    assert sq_client.api_url == "https://sq.wikipedia.org/w/api.php"


def test_client_pool_uses_override_only_for_default_lang():
    pool = MediaWikiClientPool(
        "test-agent/1.0", default_lang="en", default_api_url="https://custom.example.org/w/api.php"
    )
    assert pool.get("en").api_url == "https://custom.example.org/w/api.php"
    assert pool.get("sq").api_url == "https://sq.wikipedia.org/w/api.php"


@pytest.mark.asyncio
async def test_client_pool_aclose_closes_all():
    pool = MediaWikiClientPool("test-agent/1.0", default_lang="en")
    pool.get("en")
    pool.get("sq")
    await pool.aclose()  # should not raise


@pytest.mark.asyncio
async def test_fetch_article_success():
    with respx.mock(base_url=API_URL) as mock:
        mock.get(params={"action": "query"}).mock(
            return_value=httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "title": "Paris",
                                "revisions": [
                                    {
                                        "revid": 12345,
                                        "timestamp": "2026-01-01T00:00:00Z",
                                        "slots": {"main": {"content": "'''Paris''' is a city."}},
                                    }
                                ],
                            }
                        ]
                    }
                },
            )
        )
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        source = await client.fetch_article("Paris")
        assert source.title == "Paris"
        assert source.revid == 12345
        assert source.wikitext == "'''Paris''' is a city."
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_article_missing_raises():
    with respx.mock(base_url=API_URL) as mock:
        mock.get(params={"action": "query"}).mock(
            return_value=httpx.Response(
                200,
                json={"query": {"pages": [{"title": "Nope", "missing": True}]}},
            )
        )
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        with pytest.raises(MediaWikiError, match="does not exist"):
            await client.fetch_article("Nope")
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_category_members_with_continuation():
    with respx.mock(base_url=API_URL) as mock:
        route = mock.get(url__regex=r".*").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "query": {"categorymembers": [{"title": "Article A"}, {"title": "Article B"}]},
                        "continue": {"cmcontinue": "next-token"},
                    },
                ),
                httpx.Response(
                    200,
                    json={"query": {"categorymembers": [{"title": "Article C"}]}},
                ),
            ]
        )
        client = MediaWikiClient(API_URL, "test-agent/1.0")
        titles = await client.fetch_category_members("Test category")
        assert titles == ["Article A", "Article B", "Article C"]
        assert route.call_count == 2
        await client.aclose()
