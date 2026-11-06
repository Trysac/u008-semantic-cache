# Single-module brute-force semantic cache with injectable embedder and clock

Status: accepted (generated, pending human review)

## Context

LLM responses for repeated or paraphrased questions can be cached, but loose similarity thresholds, stale entries and missing tenant isolation return wrong or private answers. Existing frameworks are large and hard to audit. The library must stay dependency-free, deterministic under test and small enough to review.

## Decision

Implement one pure-Python module. Entries are keyed by SHA-256 of namespace, model_id and the normalized query, with an exact-key fast path before any embedder call. Vectors are L2-normalized at insertion so cosine is a dot product computed by brute force over the entries of one (namespace, model_id) group, which also fixes the expected dimension per group. TTL, LRU (OrderedDict) and counters are explicit; the clock and embedder are injected. State is guarded by a threading.Lock while the embedder runs outside it, and events are emitted after the lock is released so callbacks cannot deadlock or block the cache. Event payloads carry only type, namespace and score. All failures use one SemanticCacheError base with stable codes.

## Consequences

The code is easy to audit and test with fake clocks and embedders, but similarity search is O(n) per namespace and is documented as suitable up to about 10k entries. There is no persistence, no approximate nearest neighbour index and no async API. Concurrent misses on the same query may each call the embedder, and the last put wins. Callers must choose namespaces for tenant isolation and must avoid caching answers that depend on entities or numbers.
