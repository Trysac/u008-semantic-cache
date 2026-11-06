# Architecture

## Purpose

`trysac-semantic-cache` is a small, dependency-free Python library that caches LLM responses by embedding cosine similarity, with an exact-key fast path, TTL, tenant namespaces, LRU eviction and explicit invalidation.

## Shape

```text
caller code --> semantic_cache (public API in __init__.py)
                    |
                    +-- SemanticCache state guarded by threading.Lock
                    +-- embedder called outside the lock (EmbedderPort)
```

- **Public API**: the names listed in `__all__` of `src/semantic_cache/__init__.py`: `CacheHit`, `CacheStats`, `EmbedderPort`, `FakeEmbedder`, `SemanticCache`, `SemanticCacheError`, `__version__`.
- **State**: an ordered entry map for LRU order and TTL, plus per-(namespace, model_id) groups for similarity scans. Counters track hits, misses, evictions and expirations.
- **Errors**: invalid input raises `SemanticCacheError` (a `ValueError` subclass) with a stable `code`.
- **Logging**: a module logger with a `NullHandler`; the library never configures logging and never logs query or response text.

## Quality contract

`scripts/check.py` is the single definition of the gates; `fabrica.json` tells the factory and CI how to run it. See [TESTING.md](TESTING.md).

## Decisions

Architecture decisions are recorded in [decisions/](decisions/) as numbered ADRs.
