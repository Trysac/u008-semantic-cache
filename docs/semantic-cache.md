# semantic-cache usage

`semantic_cache` caches LLM responses by embedding cosine similarity. It has no runtime dependencies.

## Usage

```python
from semantic_cache import FakeEmbedder, SemanticCache

cache = SemanticCache(FakeEmbedder(), threshold=0.9)
cache.put("What is the capital of France?", "Paris", namespace="tenant-1")
hit = cache.get("what is the  capital of france", namespace="tenant-1")
print(hit.response if hit else "miss")
```

`FakeEmbedder` is a deterministic bag-of-words embedder meant for tests and documentation. In production pass any object with a `model_id: str` attribute and an `embed(text) -> Sequence[float]` method.

## Behaviour

- Queries are normalized (NFKC, casefold, whitespace collapsed, stripped). The embedder receives the normalized text.
- The entry key is the SHA-256 hex of `namespace`, NUL, `model_id`, NUL and the normalized query. `get` checks the exact key first (score 1.0, no embedder call); otherwise it embeds once and scans only entries of the same namespace and `model_id`.
- Vectors are L2-normalized when stored, so cosine similarity is a dot product. A hit needs `score >= threshold`; the best score wins and ties go to the most recent insertion.
- An entry is expired when `clock() - created_at >= ttl` and is purged when touched. `ttl=0` stores nothing (and drops an existing entry with the same key); `ttl=None` never expires.
- LRU: hits and puts make an entry most recent; when the cache is full the least recently used entry is evicted.
- `invalidate` takes exactly one of `key`, `tag` or `namespace` and returns the number of entries removed.
- `on_event` receives `{'type', 'namespace', 'score'}` payloads only (`hit`, `miss`, `evict`, `expire`); it never receives query or response text. Callback exceptions are logged without content and ignored.

## Limits

Brute-force similarity over pure-Python lists is fine up to about 10k entries per namespace. Persistence, vector databases, approximate nearest neighbour indexes and async are out of scope.

Defaults: `max_query_chars=8192`, `max_response_chars=1_000_000`, at most 32 tags of at most 128 characters per entry, namespaces and model ids of at most 256 characters without NUL, embeddings of at most 65,536 values.

## Security notes

- Use one namespace per user or tenant whenever responses can contain private data.
- A high threshold reduces but does not remove false hits between queries that differ only in an entity or a number. Do not cache answers that depend on such details.
- `invalidate(tag=...)` acts across all namespaces.
- The library never logs input text and never configures logging.

## Errors

Every error is a `SemanticCacheError` (a `ValueError`) with a stable `code`: `dimension_mismatch`, `invalid_embedding`, `input_too_large` or `invalid_argument`. Exceptions raised by the embedder propagate unchanged and nothing is stored.
