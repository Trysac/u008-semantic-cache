# Testing

## Running

```sh
uv run pytest --cov=semantic_cache   # tests with coverage report
uv run python scripts/check.py      # every gate, as CI and the factory run it
```

## Policy

- Branch coverage of `src/semantic_cache` must stay at or above 80 %; the gate fails below it.
- Tests exercise the public API (`import semantic_cache`) and assert observable effects: return values, raised exceptions and their codes.
- Each bug fix starts with a failing test that reproduces it.
- Tests are deterministic: they use an injected fake clock and deterministic embedders, with no network and no dependence on test order.

## Layout

- `tests/test_semantic_cache.py` is the single test module. It covers the version, the `Clock` and `MapEmbedder`, `CountingEmbedder` and `ExplodingEmbedder` test doubles, and the behaviour of `SemanticCache`: exact-key hits, cosine threshold, TTL expiry, namespace and model isolation, LRU eviction, invalidation selectors, error codes, and the absence of query or response text in events.
