# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `SemanticCache`: embedding-similarity LLM response cache with exact-key fast path, TTL, namespaces, LRU eviction, tag/key/namespace invalidation, stats and content-free events, plus `FakeEmbedder` and `SemanticCacheError`.
- Initial project structure from the factory utility template.
