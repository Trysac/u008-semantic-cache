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
| Tokens (input / output) | 476188 / 63367 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.5905 |

## Factory run u008 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2096 | 51747 | 0 | 4246 | 0.2613 |
| fix-1 | claude-sonnet-5-5 | no | 4 | 886 | 6420 | 24479 | 0.0274 |
| fix-2 | claude-sonnet-5-5 | no | 14 | 2183 | 9461 | 106603 | 0.0562 |
| fix-3 | claude-sonnet-5-5 | no | 4 | 312 | 5360 | 24068 | 0.0189 |
| fix-4 | claude-sonnet-5-5 | no | 8 | 799 | 6536 | 54326 | 0.0298 |
| audit | claude-sonnet-5-5 | no | 12 | 2687 | 33934 | 142007 | 0.1259 |
| docs | claude-haiku-5-5 | no | 118321 | 4753 | 0 | 0 | 0.0710 |

Total: $0.5905. Prices: Anthropic list prices.
