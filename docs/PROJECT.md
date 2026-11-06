# Project record

## Objective

Provide a small, auditable semantic cache for LLM responses in which the embedder, similarity threshold, TTL, namespace isolation and eviction are explicit and tested. Repeated or paraphrased questions are served from the cache instead of the model.

## Scope

- In scope: the public API documented in [README.md](../README.md), typed and tested; exact-key lookup, cosine lookup over L2-normalized vectors, TTL, namespaces, LRU eviction, invalidation by key, tag or namespace, counters and an event callback.
- Out of scope: persistence, vector databases, approximate nearest neighbour indexes, async APIs, and anything that needs runtime dependencies or I/O without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | pending measurement |
| Hours (human) | pending measurement |
| Cost (USD) | pending measurement |
