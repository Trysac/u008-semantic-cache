# Security notes

## Threat model

`trysac-semantic-cache` is a pure in-memory library without I/O.

Defended:

- **Cross-tenant leakage**: entries are keyed and searched per namespace; an entry is never returned for another namespace.
- **Cross-model confusion**: lookups only scan entries with the same `model_id`.
- **Key forgery across fields**: NUL characters are rejected in namespace and model_id, and the key is a SHA-256 over namespace, model_id and normalized query.
- **Oversized or malformed input**: query, response, tag, namespace, embedding length and values are validated; invalid input raises `SemanticCacheError` with a stable code.
- **Content leakage through events and logs**: `on_event` payloads contain only type, namespace and score; the library never logs query or response text.
- **Concurrent access**: all cache state is accessed under a `threading.Lock`.

Not defended:

- **Semantic false hits**: a threshold reduces but does not remove hits between queries that differ only in an entity or a number. Do not cache answers that depend on such details.
- **Data at rest**: nothing is persisted; memory is not encrypted or wiped.
- **Shared namespaces**: callers that place different users in one namespace share their cached responses.
- **Embedder trust**: the embedder is called with the normalized query text and its output is trusted only after validation for dimension, finiteness and non-zero norm.
- **Denial of service by size**: limits are per call (`max_query_chars`, `max_response_chars`, 65 536 dimensions); the total number of entries is bounded by `max_entries`, and similarity scans are linear in the namespace size.

## Publishing

Releases are published to PyPI with trusted publishing (OIDC) when available, never with a long-lived token stored in the repository.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).
