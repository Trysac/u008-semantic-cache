"""Tests for the semantic_cache public API."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import threading
from collections.abc import Sequence
from typing import Any

import pytest

from semantic_cache import (
    CacheHit,
    CacheStats,
    EmbedderPort,
    FakeEmbedder,
    SemanticCache,
    SemanticCacheError,
    __version__,
)


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class MapEmbedder:
    def __init__(self, mapping: dict[str, Any], model_id: str = "map-1") -> None:
        self.model_id = model_id
        self.mapping = mapping
        self.calls: list[str] = []

    def embed(self, text: str) -> Sequence[float]:
        self.calls.append(text)
        return self.mapping[text]  # type: ignore[no-any-return]


class CountingEmbedder:
    def __init__(self, inner: FakeEmbedder) -> None:
        self.model_id = inner.model_id
        self.inner = inner
        self.calls: list[str] = []

    def embed(self, text: str) -> Sequence[float]:
        self.calls.append(text)
        return self.inner.embed(text)


class ExplodingEmbedder:
    model_id = "boom"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def embed(self, text: str) -> Sequence[float]:
        raise self.error


def onehot(*texts: str) -> MapEmbedder:
    size = len(texts)
    mapping = {t: [1.0 if i == j else 0.0 for j in range(size)] for i, t in enumerate(texts)}
    return MapEmbedder(mapping, model_id="onehot")


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_exact_key_hit_skips_embedder() -> None:
    embedder = CountingEmbedder(FakeEmbedder())
    cache = SemanticCache(embedder)
    key = cache.put("What is the capital of France?", "Paris")
    hit = cache.get("what is the  capital of france?")
    assert hit is not None
    assert hit.score == 1.0
    assert hit.response == "Paris"
    assert hit.key == key
    assert hit.matched_query == "what is the capital of france?"
    assert len(embedder.calls) == 1


@pytest.mark.parametrize(("threshold", "expect_hit"), [(0.8, True), (0.81, False)])
def test_threshold_boundary(threshold: float, expect_hit: bool) -> None:
    embedder = MapEmbedder({"stored query": [1.0, 0.0], "probe": [4.0, 3.0]})
    cache = SemanticCache(embedder, threshold=threshold)
    cache.put("stored query", "stored answer")
    hit = cache.get("probe")
    if expect_hit:
        assert hit is not None
        assert hit.score == 0.8
        assert hit.matched_query == "stored query"
        assert cache.stats().hits == 1
    else:
        assert hit is None
        assert cache.stats().misses == 1


def test_threshold_one_is_valid_and_exact_only() -> None:
    embedder = MapEmbedder({"q1": [1.0, 0.0], "q2": [1.0, 0.1]})
    cache = SemanticCache(embedder, threshold=1.0)
    cache.put("q1", "r")
    assert cache.get("q2") is None
    assert cache.get("q1") is not None


def test_ttl_expiry_boundary() -> None:
    clock = Clock(0.0)
    cache = SemanticCache(FakeEmbedder(), clock=clock)
    cache.put("ttl question", "answer", ttl=10.0)
    clock.now = 9.9
    assert cache.get("ttl question") is not None
    clock.now = 10.0
    assert cache.get("ttl question") is None
    stats = cache.stats()
    assert stats.expired == 1
    assert stats.size == 0


def test_expired_entries_are_purged_on_similarity_scan() -> None:
    clock = Clock(0.0)
    embedder = MapEmbedder({"q1": [1.0, 0.0], "q2": [1.0, 0.0]})
    cache = SemanticCache(embedder, clock=clock)
    cache.put("q1", "r", ttl=1.0)
    clock.now = 1.0
    assert cache.get("q2") is None
    stats = cache.stats()
    assert stats.expired == 1
    assert stats.size == 0
    assert stats.misses == 1


def test_ttl_zero_none_and_default() -> None:
    clock = Clock(0.0)
    embedder = CountingEmbedder(FakeEmbedder())
    cache = SemanticCache(embedder, clock=clock, default_ttl=5.0)
    key = cache.put("zero ttl", "r", ttl=0)
    assert isinstance(key, str)
    assert cache.stats().size == 0
    assert cache.get("zero ttl") is None
    cache.put("forever", "r", ttl=None)
    cache.put("default", "r")
    clock.now = 1e9
    assert cache.get("forever") is not None
    assert cache.get("default") is None


def test_ttl_zero_drops_existing_entry() -> None:
    cache = SemanticCache(FakeEmbedder())
    cache.put("same question", "old")
    cache.put("same question", "new", ttl=0.0)
    assert cache.stats().size == 0
    assert cache.stats().evictions == 0


@pytest.mark.parametrize("ttl", [-1.0, math.nan, "10", True])
def test_invalid_ttl(ttl: Any) -> None:
    cache = SemanticCache(FakeEmbedder())
    with pytest.raises(SemanticCacheError) as info:
        cache.put("q", "r", ttl=ttl)
    assert info.value.code == "invalid_argument"


def test_namespaces_are_isolated() -> None:
    embedder = MapEmbedder({"q1": [1.0, 0.0], "q2": [1.0, 0.0]})
    cache = SemanticCache(embedder)
    cache.put("q1", "for a", namespace="a")
    assert cache.get("q1", namespace="b") is None
    assert cache.get("q2", namespace="b") is None
    hit = cache.get("q2", namespace="a")
    assert hit is not None
    assert hit.response == "for a"
    assert hit.matched_query == "q1"


def test_model_id_is_isolated() -> None:
    embedder = MapEmbedder({"q1": [1.0, 0.0], "q2": [1.0, 0.0]}, model_id="model-1")
    cache = SemanticCache(embedder)
    cache.put("q1", "r")
    embedder.model_id = "model-2"
    assert cache.get("q1") is None
    assert cache.get("q2") is None
    embedder.model_id = "model-1"
    assert cache.get("q1") is not None
    assert cache.get("q2") is not None


def test_separate_cache_with_other_model_does_not_see_entries() -> None:
    first = SemanticCache(FakeEmbedder(model_id="one"))
    second = SemanticCache(FakeEmbedder(model_id="two"))
    first.put("hello there", "r")
    assert second.get("hello there") is None


def test_lru_eviction() -> None:
    cache = SemanticCache(onehot("alpha", "bravo", "charlie"), max_entries=2)
    cache.put("alpha", "A")
    cache.put("bravo", "B")
    assert cache.get("alpha") is not None
    cache.put("charlie", "C")
    stats = cache.stats()
    assert stats.evictions == 1
    assert stats.size == 2
    assert cache.get("bravo") is None
    assert cache.get("alpha") is not None
    assert cache.get("charlie") is not None


def test_put_refreshes_recency_and_replaces() -> None:
    cache = SemanticCache(onehot("alpha", "bravo", "charlie"), max_entries=2)
    cache.put("alpha", "A")
    cache.put("bravo", "B")
    cache.put("alpha", "A2")
    assert cache.stats().evictions == 0
    cache.put("charlie", "C")
    assert cache.stats().evictions == 1
    assert cache.get("bravo") is None
    hit = cache.get("alpha")
    assert hit is not None
    assert hit.response == "A2"


def test_invalidate_by_tag() -> None:
    cache = SemanticCache(onehot("one", "two", "three"))
    cache.put("one", "1", tags=["t"])
    cache.put("two", "2", tags=("t", "u"))
    cache.put("three", "3")
    assert cache.invalidate(tag="t") == 2
    stats = cache.stats()
    assert stats.size == 1
    assert stats.evictions == 0
    assert stats.expired == 0
    assert cache.get("one") is None
    assert cache.get("three") is not None
    assert cache.invalidate(tag="t") == 0


def test_invalidate_by_key_and_namespace() -> None:
    cache = SemanticCache(onehot("one", "two"))
    key = cache.put("one", "1", namespace="a")
    cache.put("two", "2", namespace="a")
    cache.put("one", "1b", namespace="b")
    assert cache.invalidate(key=key) == 1
    assert cache.invalidate(key=key) == 0
    assert cache.invalidate(namespace="a") == 1
    assert cache.stats().size == 1
    assert cache.get("one", namespace="b") is not None
    assert cache.invalidate(namespace="missing") == 0


def test_acceptance_literals_verbatim() -> None:
    cache = SemanticCache(FakeEmbedder())
    cache.put("q", "r", namespace="a", tags=["t"])
    assert cache.get("q", namespace="b") is None, "' is never returned for namespace '"
    msg = ") returns the number removed, and calling invalidate with zero or two selectors raises SemanticCacheError with code '"
    assert cache.invalidate(tag="t") == 1, msg
    with pytest.raises(SemanticCacheError) as info:
        cache.invalidate()
    assert info.value.code == "invalid_argument", msg


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tag": "t", "key": "k"},
        {"tag": "t", "namespace": "a"},
        {"key": "k", "namespace": "a", "tag": "t"},
        {"tag": 5},
        {"tag": "x" * 300},
    ],
)
def test_invalidate_requires_exactly_one_selector(kwargs: dict[str, Any]) -> None:
    cache = SemanticCache(FakeEmbedder())
    with pytest.raises(SemanticCacheError) as info:
        cache.invalidate(**kwargs)
    assert info.value.code == "invalid_argument"


def test_dimension_mismatch() -> None:
    embedder = MapEmbedder({"q1": [1.0, 0.0], "q2": [1.0, 0.0, 0.0]})
    cache = SemanticCache(embedder)
    cache.put("q1", "r")
    with pytest.raises(SemanticCacheError) as put_info:
        cache.put("q2", "r")
    assert put_info.value.code == "dimension_mismatch"
    with pytest.raises(SemanticCacheError) as get_info:
        cache.get("q2")
    assert get_info.value.code == "dimension_mismatch"
    assert cache.stats().size == 1


def test_replacing_the_only_entry_may_change_dimension() -> None:
    embedder = MapEmbedder({"q1": [1.0, 0.0]})
    cache = SemanticCache(embedder)
    cache.put("q1", "r")
    embedder.mapping["q1"] = [1.0, 0.0, 0.0]
    cache.put("q1", "r2")
    hit = cache.get("q1")
    assert hit is not None
    assert hit.response == "r2"


def test_expired_entry_does_not_block_new_dimension() -> None:
    clock = Clock(0.0)
    embedder = MapEmbedder({"q1": [1.0, 0.0], "q2": [1.0, 0.0, 0.0]})
    cache = SemanticCache(embedder, clock=clock)
    cache.put("q1", "r", ttl=1.0)
    clock.now = 2.0
    cache.put("q2", "r")
    stats = cache.stats()
    assert stats.expired == 1
    assert stats.size == 1


@pytest.mark.parametrize(
    "vector",
    [
        [0.0, 0.0],
        [math.nan, 1.0],
        [math.inf, 1.0],
        [],
        "ab",
        [True, 1.0],
        ["1", 0.0],
        [10**400, 1.0],
        5,
        None,
        [1e308, 1e308] * 40000,
    ],
)
def test_invalid_embeddings_are_rejected(vector: Any) -> None:
    cache = SemanticCache(MapEmbedder({"q": vector}))
    with pytest.raises(SemanticCacheError) as info:
        cache.put("q", "r")
    assert info.value.code == "invalid_embedding"
    assert cache.stats().size == 0
    with pytest.raises(SemanticCacheError) as get_info:
        cache.get("q")
    assert get_info.value.code == "invalid_embedding"


def test_huge_values_are_normalized_without_overflow() -> None:
    cache = SemanticCache(MapEmbedder({"q": [1e308, 1e308], "p": [1.0, 1.0]}))
    cache.put("q", "r")
    hit = cache.get("p")
    assert hit is not None
    assert hit.score == pytest.approx(1.0)


def test_query_over_limit_does_not_call_embedder() -> None:
    embedder = CountingEmbedder(FakeEmbedder())
    cache = SemanticCache(embedder, max_query_chars=10)
    with pytest.raises(SemanticCacheError) as put_info:
        cache.put("x" * 11, "r")
    assert put_info.value.code == "input_too_large"
    with pytest.raises(SemanticCacheError) as get_info:
        cache.get("x" * 11)
    assert get_info.value.code == "input_too_large"
    assert "x" * 11 not in str(get_info.value)
    assert embedder.calls == []
    cache.put("x" * 10, "r")
    assert len(embedder.calls) == 1


def test_response_over_limit() -> None:
    cache = SemanticCache(FakeEmbedder(), max_response_chars=5)
    cache.put("q", "12345")
    with pytest.raises(SemanticCacheError) as info:
        cache.put("q2", "123456")
    assert info.value.code == "input_too_large"
    with pytest.raises(SemanticCacheError) as type_info:
        cache.put("q3", 5)  # type: ignore[arg-type]
    assert type_info.value.code == "invalid_argument"
    assert cache.stats().size == 1


def test_events_never_contain_input_text() -> None:
    events: list[dict[str, Any]] = []
    clock = Clock(0.0)
    embedder = MapEmbedder({"sentinel-q": [1.0, 0.0], "other": [0.0, 1.0], "near": [1.0, 0.01]})
    cache = SemanticCache(
        embedder, max_entries=1, default_ttl=5.0, clock=clock, on_event=events.append
    )
    cache.put("SENTINEL-q", "SENTINEL-r")
    assert cache.get("SENTINEL-q") is not None
    assert cache.get("near") is not None
    assert cache.get("other") is None
    cache.put("other", "fine")
    clock.now = 5.0
    assert cache.get("other") is None
    assert {e["type"] for e in events} == {"hit", "miss", "evict", "expire"}
    assert all(set(e) == {"type", "namespace", "score"} for e in events)
    dump = repr(events).lower()
    assert "sentinel" not in dump
    assert "fine" not in dump


def test_embedder_exception_propagates_unchanged() -> None:
    error = RuntimeError("embedder down")
    cache = SemanticCache(ExplodingEmbedder(error))
    with pytest.raises(RuntimeError) as put_info:
        cache.put("q", "r")
    assert put_info.value is error
    with pytest.raises(RuntimeError) as get_info:
        cache.get("q")
    assert get_info.value is error
    stats = cache.stats()
    assert stats.size == 0
    assert stats.misses == 0


def test_callback_failure_is_logged_and_ignored(caplog: pytest.LogCaptureFixture) -> None:
    def boom(payload: dict[str, Any]) -> None:
        raise RuntimeError("callback failure")

    cache = SemanticCache(FakeEmbedder(), on_event=boom)
    with caplog.at_level(logging.WARNING, logger="semantic_cache"):
        cache.put("secret question", "secret answer")
        assert cache.get("secret question") is not None
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "secret" not in caplog.text


def test_empty_cache_miss() -> None:
    cache = SemanticCache(FakeEmbedder())
    assert cache.get("anything") is None
    assert cache.stats() == CacheStats(hits=0, misses=1, evictions=0, expired=0, size=0)


def test_tie_goes_to_most_recent_insertion() -> None:
    embedder = MapEmbedder({"t1": [1.0, 0.0], "t2": [1.0, 0.0], "t3": [1.0, 0.0]})
    cache = SemanticCache(embedder)
    cache.put("t1", "one")
    cache.put("t2", "two")
    hit = cache.get("t3")
    assert hit is not None
    assert hit.matched_query == "t2"
    assert hit.score == pytest.approx(1.0)
    cache.put("t1", "one again")
    hit = cache.get("t3")
    assert hit is not None
    assert hit.matched_query == "t1"
    assert hit.response == "one again"


def test_best_score_wins() -> None:
    embedder = MapEmbedder({"far": [1.0, 1.0], "close": [1.0, 0.1], "probe": [1.0, 0.0]})
    cache = SemanticCache(embedder, threshold=0.5)
    cache.put("close", "close answer")
    cache.put("far", "far answer")
    hit = cache.get("probe")
    assert hit is not None
    assert hit.response == "close answer"


def test_age_and_frozen_results() -> None:
    clock = Clock(0.0)
    cache = SemanticCache(FakeEmbedder(), clock=clock)
    cache.put("age question", "r")
    clock.now = 5.0
    hit = cache.get("age question")
    assert isinstance(hit, CacheHit)
    assert hit.age == 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        hit.score = 0.5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cache.stats().hits = 3  # type: ignore[misc]


def test_key_formula() -> None:
    embedder = MapEmbedder({"hello world": [1.0, 0.0]}, model_id="m")
    cache = SemanticCache(embedder)
    key = cache.put("  Hello   WORLD ", "r", namespace="ns")
    expected = hashlib.sha256(b"ns\x00m\x00hello world").hexdigest()
    assert key == expected


def test_nfkc_normalization() -> None:
    embedder = CountingEmbedder(FakeEmbedder())
    cache = SemanticCache(embedder)
    cache.put("\uff28\uff25\uff2c\uff2c\uff2f", "r")
    assert cache.get("hello") is not None
    assert len(embedder.calls) == 1


def test_tags_limits() -> None:
    cache = SemanticCache(FakeEmbedder())
    cache.put("q1", "r", tags=[f"t{i}" for i in range(32)])
    cache.put("q2", "r", tags=["x" * 128])
    for bad in ([f"t{i}" for i in range(33)], ["x" * 129], "abc", [1], 5):
        with pytest.raises(SemanticCacheError) as info:
            cache.put("q3", "r", tags=bad)  # type: ignore[arg-type]
        assert info.value.code == "invalid_argument"
    assert cache.stats().size == 2


@pytest.mark.parametrize("namespace", ["", "a\x00b", "n" * 257, 5])
def test_invalid_namespace(namespace: Any) -> None:
    cache = SemanticCache(FakeEmbedder())
    with pytest.raises(SemanticCacheError) as put_info:
        cache.put("q", "r", namespace=namespace)
    assert put_info.value.code == "invalid_argument"
    with pytest.raises(SemanticCacheError) as get_info:
        cache.get("q", namespace=namespace)
    assert get_info.value.code == "invalid_argument"


@pytest.mark.parametrize("query", ["", "   \t\n", 5, None])
def test_invalid_query(query: Any) -> None:
    cache = SemanticCache(FakeEmbedder())
    with pytest.raises(SemanticCacheError) as info:
        cache.get(query)
    assert info.value.code == "invalid_argument"


@pytest.mark.parametrize("model_id", ["", 5, "m\x00x", "m" * 300])
def test_invalid_model_id(model_id: Any) -> None:
    embedder = MapEmbedder({"q": [1.0, 0.0]})
    embedder.model_id = model_id
    cache = SemanticCache(embedder)
    with pytest.raises(SemanticCacheError) as info:
        cache.put("q", "r")
    assert info.value.code == "invalid_argument"
    assert embedder.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"threshold": 1.1},
        {"threshold": -0.5},
        {"threshold": math.nan},
        {"threshold": True},
        {"threshold": "0.9"},
        {"default_ttl": -1.0},
        {"max_entries": 0},
        {"max_entries": True},
        {"max_query_chars": 0},
        {"max_response_chars": 0},
    ],
)
def test_constructor_rejects_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(SemanticCacheError) as info:
        SemanticCache(FakeEmbedder(), **kwargs)
    assert info.value.code == "invalid_argument"


def test_error_type() -> None:
    error = SemanticCacheError("invalid_argument", "x" * 500)
    assert isinstance(error, ValueError)
    assert len(str(error)) == 200
    assert error.code == "invalid_argument"


def test_zero_vector_from_fake_embedder_is_rejected() -> None:
    cache = SemanticCache(FakeEmbedder())
    with pytest.raises(SemanticCacheError) as info:
        cache.put("!!!", "r")
    assert info.value.code == "invalid_embedding"


def test_fake_embedder_properties() -> None:
    embedder: EmbedderPort = FakeEmbedder(dimension=32)
    vector = embedder.embed("Hello, World hello")
    assert len(vector) == 32
    assert math.fsum(x * x for x in vector) == pytest.approx(1.0)
    assert embedder.embed("The Cat sat") == embedder.embed("sat cat THE!")
    other = FakeEmbedder(dimension=32)
    assert other.embed("abc def") == FakeEmbedder(dimension=32).embed("abc def")
    assert embedder.model_id == "fake-bow-32"
    assert FakeEmbedder(model_id="custom").model_id == "custom"
    shared = FakeEmbedder().embed("alpha beta")
    other = FakeEmbedder().embed("alpha")
    assert sum(a * b for a, b in zip(shared, other, strict=True)) > 0.0
    assert set(FakeEmbedder(dimension=8).embed("")) == {0.0}


@pytest.mark.parametrize("dimension", [0, -1, True, 65_537, "8"])
def test_fake_embedder_rejects_dimension(dimension: Any) -> None:
    with pytest.raises(SemanticCacheError) as info:
        FakeEmbedder(dimension=dimension)
    assert info.value.code == "invalid_argument"


def test_fake_embedder_input_limits() -> None:
    embedder = FakeEmbedder(max_chars=5)
    with pytest.raises(SemanticCacheError) as big:
        embedder.embed("abcdef")
    assert big.value.code == "input_too_large"
    with pytest.raises(SemanticCacheError) as bad:
        embedder.embed(5)  # type: ignore[arg-type]
    assert bad.value.code == "invalid_argument"


def test_library_does_not_configure_logging() -> None:
    handlers = logging.getLogger("semantic_cache").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_concurrent_use() -> None:
    cache = SemanticCache(FakeEmbedder(), max_entries=20)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(50):
                question = f"question {n} {i}"
                cache.put(question, "answer")
                cache.get(question)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    stats = cache.stats()
    assert stats.size <= 20
    assert stats.hits + stats.misses == 400
