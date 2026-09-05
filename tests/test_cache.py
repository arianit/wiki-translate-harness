from pathlib import Path

from wiki_translation_harness.cache import TranslationCache, compute_key


def test_key_differs_by_model():
    k1 = compute_key("model-a", "en", "sq", "hello")
    k2 = compute_key("model-b", "en", "sq", "hello")
    assert k1 != k2


def test_key_differs_by_langs():
    k1 = compute_key("m", "en", "sq", "hello")
    k2 = compute_key("m", "en", "fr", "hello")
    assert k1 != k2


def test_key_differs_by_skill_hash():
    # editing the skill file or the harness's invocation framing must
    # invalidate old cache entries rather than silently reusing them
    k1 = compute_key("m", "en", "sq", "hello", skill_hash="hash-a")
    k2 = compute_key("m", "en", "sq", "hello", skill_hash="hash-b")
    assert k1 != k2


def test_key_stable_for_same_input():
    assert compute_key("m", "en", "sq", "hello") == compute_key("m", "en", "sq", "hello")


def test_set_and_get(tmp_path: Path):
    cache = TranslationCache(tmp_path / "cache.sqlite3")
    key = compute_key("m", "en", "sq", "hello")
    assert cache.get(key) is None
    cache.set(key, "m", "en", "sq", "hello", "përshëndetje")
    assert cache.get(key) == "përshëndetje"
    cache.close()


def test_cache_persists_across_reopen(tmp_path: Path):
    db_path = tmp_path / "cache.sqlite3"
    key = compute_key("m", "en", "sq", "hello")

    cache1 = TranslationCache(db_path)
    cache1.set(key, "m", "en", "sq", "hello", "përshëndetje")
    cache1.close()

    cache2 = TranslationCache(db_path)
    assert cache2.get(key) == "përshëndetje"
    cache2.close()


def test_cross_model_no_collision(tmp_path: Path):
    cache = TranslationCache(tmp_path / "cache.sqlite3")
    key_a = compute_key("model-a", "en", "sq", "hello")
    key_b = compute_key("model-b", "en", "sq", "hello")
    cache.set(key_a, "model-a", "en", "sq", "hello", "translation-a")
    assert cache.get(key_b) is None
    assert cache.get(key_a) == "translation-a"
    cache.close()
