# semantic-cache

[![PyPI version](https://img.shields.io/pypi/v/trysac-semantic-cache)](https://pypi.org/project/trysac-semantic-cache/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-semantic-cache)](LICENSE)
[![CI](https://github.com/Trysac/u008-semantic-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u008-semantic-cache/actions/workflows/ci.yml)

Cache LLM responses by embedding cosine similarity with an exact-key fast path, TTL, tenant namespaces, LRU eviction and explicit invalidation

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-semantic-cache
# or
uv add trysac-semantic-cache
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import semantic_cache

# normalize() trims surrounding whitespace and lower-cases the text.
print(semantic_cache.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    semantic_cache.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
