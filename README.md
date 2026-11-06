# semantic-cache

[![PyPI version](https://img.shields.io/pypi/v/trysac-semantic-cache)](https://pypi.org/project/trysac-semantic-cache/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-semantic-cache)](LICENSE)
[![CI](https://github.com/Trysac/u008-semantic-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u008-semantic-cache/actions/workflows/ci.yml)

Cache LLM responses by embedding cosine similarity with an exact-key fast path, TTL, tenant namespaces, LRU eviction and explicit invalidation

## Problem

Repeated or paraphrased questions sent to an LLM cost money and latency on every call. Exact-match caches miss paraphrases, while similarity caches with a loose threshold, stale entries or no tenant isolation return wrong or private answers. This library provides a small, auditable semantic cache in which the embedder, threshold, TTL, namespace isolation and eviction are explicit and tested.

## Install

```sh
pip install trysac-semantic-cache
# or
uv add trysac-semantic-cache
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
from semantic_cache import FakeEmbedder, SemanticCache

cache = SemanticCache(FakeEmbedder(), threshold=0.9)
cache.put("What is the capital of France?", "Paris", namespace="tenant-1")
hit = cache.get("what is the  capital of france", namespace="tenant-1")
print(hit.response if hit else "miss")  # Paris
```

## Public API

- `EmbedderPort` (Protocol): `model_id: str` and `embed(text) -> Sequence[float]`. Vectors are L2-normalized by the cache.
- `FakeEmbedder(dimension=64, *, model_id=None, max_chars=65_536)`: deterministic bag-of-words embedder for tests and examples.
- `SemanticCache(embedder, *, threshold=0.92, default_ttl=3600.0, max_entries=1000, max_query_chars=8192, max_response_chars=1_000_000, clock=time.monotonic, on_event=None)`.
- `SemanticCache.get(query, *, namespace="default") -> CacheHit | None`: exact normalized-key check first (score 1.0, no embedder call), then cosine search over the same namespace and model.
- `SemanticCache.put(query, response, *, namespace="default", ttl=..., tags=()) -> str`: returns the entry key. `ttl=0` stores nothing; `ttl=None` never expires.
- `SemanticCache.invalidate(*, key=None, tag=None, namespace=None) -> int`: exactly one selector must be given.
- `SemanticCache.stats() -> CacheStats`.
- `CacheHit(response, score, matched_query, key, age)` and `CacheStats(hits, misses, evictions, expired, size)`: frozen dataclasses.
- `SemanticCacheError(ValueError)` with `code` in `dimension_mismatch`, `invalid_embedding`, `invalid_argument`, `input_too_large`.

Behaviour and limits:

- A hit requires cosine score >= `threshold`, which must be in (0, 1]. The best score wins; ties go to the most recent insertion.
- Queries are normalized with NFKC, casefolding and whitespace collapsing. Limit: `max_query_chars`; responses: `max_response_chars`.
- An entry expires when `clock() - created_at >= ttl`. Expired entries are purged when touched.
- Each entry holds at most 32 tags of at most 128 characters.
- LRU eviction when `max_entries` is exceeded, counted across all namespaces.
- Similarity search is a brute-force scan in pure Python; intended up to about 10k entries per namespace.
- `on_event` receives only `{'type', 'namespace', 'score'}` payloads with type `hit`, `miss`, `evict` or `expire`.

## Why not GPTCache

GPTCache is a large framework. This library is a single module with an injectable clock and embedder, which makes it easy to audit.

## Security

See [docs/SECURITY.md](docs/SECURITY.md). Use one namespace per user or tenant when responses can contain private data. A high threshold does not remove false hits between queries that differ only in an entity or number.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
