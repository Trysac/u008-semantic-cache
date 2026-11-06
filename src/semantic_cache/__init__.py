"""Semantic cache for LLM responses keyed by embedding cosine similarity.

A response stored with :meth:`SemanticCache.put` is returned by :meth:`SemanticCache.get` for the
same normalized query (exact-key fast path, no embedder call) or for a paraphrase whose embedding
has cosine similarity >= ``threshold``. Entries live in namespaces, expire by TTL, are evicted in
LRU order and can be invalidated by key, tag or namespace.

Similarity search is a brute-force dot product over pure-Python float lists; it is fine up to about
10k entries per namespace. Persistence, vector databases, approximate nearest neighbour indexes
and async are out of scope.

Security notes:

* Use one namespace per user or tenant whenever responses can contain private data. Entries are
  never returned across namespaces.
* A high threshold reduces but does not remove false hits between queries that differ only in an
  entity or a number. Do not cache answers that depend on such details.
* The library never logs query or response text and never configures logging.
"""

from __future__ import annotations

import hashlib
import logging
import math
import numbers
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from types import EllipsisType
from typing import Any, Literal, Protocol, TypeGuard

__all__ = [
    "CacheHit",
    "CacheStats",
    "EmbedderPort",
    "FakeEmbedder",
    "SemanticCache",
    "SemanticCacheError",
    "__version__",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorCode = Literal["dimension_mismatch", "invalid_embedding", "invalid_argument", "input_too_large"]

_MAX_DIMENSION = 65_536
_MAX_TAGS = 32
_MAX_TAG_CHARS = 128
_MAX_NAMESPACE_CHARS = 256
_MAX_MODEL_ID_CHARS = 256
_DIM_MSG = "embedding dimension differs from cached entries"
_WORD = re.compile(r"\w+")


class SemanticCacheError(ValueError):
    """Base error of the library.

    ``code`` is a stable machine-readable identifier: ``dimension_mismatch``,
    ``invalid_embedding``, ``input_too_large`` or ``invalid_argument``. Messages never contain
    cached text and are at most 200 characters long.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message[:200])
        self.code: ErrorCode = code


class EmbedderPort(Protocol):
    """Port for the component that turns text into a vector."""

    model_id: str

    def embed(self, text: str) -> Sequence[float]:
        """Return the embedding of ``text``."""


@dataclass(frozen=True, slots=True)
class CacheHit:
    """A cache hit: the stored response and how it was matched."""

    response: str
    score: float
    matched_query: str
    key: str
    age: float


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Cumulative counters plus the current number of entries."""

    hits: int
    misses: int
    evictions: int
    expired: int
    size: int


@dataclass(frozen=True, slots=True)
class _Entry:
    key: str
    namespace: str
    model_id: str
    query: str
    response: str
    vector: tuple[float, ...]
    created_at: float
    ttl: float | None
    tags: tuple[str, ...]
    seq: int


def _is_real(value: object) -> TypeGuard[numbers.Real]:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class FakeEmbedder:
    """Deterministic bag-of-words embedder for tests and documentation.

    Each lowercase word increments one of ``dimension`` buckets chosen from its SHA-256 digest; the
    vector is L2-normalized. Texts with no word characters give the zero vector, which the cache
    rejects as ``invalid_embedding``.
    """

    def __init__(
        self,
        dimension: int = 64,
        *,
        model_id: str | None = None,
        max_chars: int = 65_536,
    ) -> None:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 1 <= dimension <= _MAX_DIMENSION
        ):
            raise SemanticCacheError("invalid_argument", "dimension out of range")
        self.dimension = dimension
        self.model_id = model_id if model_id is not None else f"fake-bow-{dimension}"
        self._max_chars = max_chars

    def embed(self, text: str) -> tuple[float, ...]:
        """Return the normalized bag-of-words vector of ``text``."""
        if not isinstance(text, str):
            raise SemanticCacheError("invalid_argument", "text must be a string")
        if len(text) > self._max_chars:
            raise SemanticCacheError("input_too_large", "text exceeds the size limit")
        counts = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode("utf-8", "surrogatepass")).digest()
            counts[int.from_bytes(digest[:8], "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(c * c for c in counts))
        if norm == 0.0:
            return tuple(counts)
        return tuple(c / norm for c in counts)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _make_key(namespace: str, model_id: str, normalized: str) -> str:
    raw = "\x00".join((namespace, model_id, normalized)).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()


def _event(kind: str, namespace: str, score: float | None) -> dict[str, Any]:
    return {"type": kind, "namespace": namespace, "score": score}


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    total = sum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, total))


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticCacheError("invalid_argument", f"{name} must be a positive integer")
    return value


def _validate_ttl(ttl: object) -> float | None:
    if ttl is None:
        return None
    if not _is_real(ttl):
        raise SemanticCacheError("invalid_argument", "ttl must be a number or None")
    value = float(ttl)
    if math.isnan(value) or value < 0.0:
        raise SemanticCacheError("invalid_argument", "ttl must be >= 0")
    return value


def _validate_tags(tags: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Iterable):
        raise SemanticCacheError("invalid_argument", "tags must be a sequence of strings")
    items = list(islice(tags, _MAX_TAGS + 1))
    if len(items) > _MAX_TAGS:
        raise SemanticCacheError("invalid_argument", f"at most {_MAX_TAGS} tags are allowed")
    for tag in items:
        if not isinstance(tag, str) or len(tag) > _MAX_TAG_CHARS:
            raise SemanticCacheError("invalid_argument", "tags must be short strings")
    return tuple(dict.fromkeys(items))


def _check_namespace(namespace: str) -> None:
    if not isinstance(namespace, str) or not namespace:
        raise SemanticCacheError("invalid_argument", "namespace must be a non-empty string")
    if len(namespace) > _MAX_NAMESPACE_CHARS or "\x00" in namespace:
        raise SemanticCacheError("invalid_argument", "namespace is too long or has a NUL")


def _unit_vector(raw: object) -> tuple[float, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise SemanticCacheError("invalid_embedding", "embedding must be a sequence of numbers")
    items = list(islice(raw, _MAX_DIMENSION + 1))
    if not items or len(items) > _MAX_DIMENSION:
        raise SemanticCacheError("invalid_embedding", "embedding has an unsupported length")
    values: list[float] = []
    for item in items:
        if not _is_real(item):
            raise SemanticCacheError("invalid_embedding", "embedding values must be numbers")
        try:
            number = float(item)
        except OverflowError:
            raise SemanticCacheError("invalid_embedding", "embedding value overflows") from None
        if not math.isfinite(number):
            raise SemanticCacheError("invalid_embedding", "embedding values must be finite")
        values.append(number)
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0.0:
        raise SemanticCacheError("invalid_embedding", "embedding must have a non-zero norm")
    return tuple(v / norm for v in values)


class SemanticCache:
    """In-memory semantic cache. All state access holds a ``threading.Lock``.

    The embedder is called outside the lock so a slow embedder never blocks other callers.
    Use one namespace per user or tenant when responses can be private, and do not cache answers
    that depend on entities or numbers: a high threshold reduces but does not remove false hits
    between queries that differ only in such details.

    Args:
        embedder: object with ``model_id`` and ``embed(text)``.
        threshold: minimum cosine similarity for a hit, in ``(0, 1]``.
        default_ttl: seconds an entry lives; ``None`` means forever.
        max_entries: LRU capacity across all namespaces.
        max_query_chars: maximum raw query length.
        max_response_chars: maximum response length.
        clock: monotonic time source in seconds (inject a fake in tests).
        on_event: callback receiving ``{'type', 'namespace', 'score'}`` payloads only.
    """

    def __init__(
        self,
        embedder: EmbedderPort,
        *,
        threshold: float = 0.92,
        default_ttl: float | None = 3600.0,
        max_entries: int = 1000,
        max_query_chars: int = 8192,
        max_response_chars: int = 1_000_000,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not _is_real(threshold) or not 0.0 < float(threshold) <= 1.0:
            raise SemanticCacheError("invalid_argument", "threshold must be in (0, 1]")
        self._embedder = embedder
        self._threshold = float(threshold)
        self._default_ttl = _validate_ttl(default_ttl)
        self._max_entries = _positive_int(max_entries, "max_entries")
        self._max_query_chars = _positive_int(max_query_chars, "max_query_chars")
        self._max_response_chars = _positive_int(max_response_chars, "max_response_chars")
        self._clock = clock
        self._on_event = on_event
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._groups: dict[tuple[str, str], dict[str, _Entry]] = {}
        self._seq = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired_count = 0

    def get(self, query: str, *, namespace: str = "default") -> CacheHit | None:
        """Return the best entry with score >= threshold in ``namespace``, or ``None``.

        The exact normalized query is checked first (score 1.0, no embedder call). Otherwise the
        query is embedded once and only entries of the same namespace and model are scanned.
        The best score wins; ties go to the most recent insertion.

        Raises:
            SemanticCacheError: on invalid input or an invalid embedding. Exceptions raised by
                the embedder propagate unchanged.
        """
        normalized = self._prepare_query(query, namespace)
        model_id = self._current_model_id()
        key = _make_key(namespace, model_id, normalized)
        events: list[dict[str, Any]] = []
        try:
            with self._lock:
                exact = self._lookup_exact(key, namespace, events)
        finally:
            self._emit(events)
        if exact is not None:
            return exact
        vector = self._embed(normalized)
        events = []
        try:
            with self._lock:
                return self._lookup_similar(vector, namespace, model_id, events)
        finally:
            self._emit(events)

    def put(
        self,
        query: str,
        response: str,
        *,
        namespace: str = "default",
        ttl: float | None | EllipsisType = ...,
        tags: Sequence[str] = (),
    ) -> str:
        """Store ``response`` for ``query`` and return the entry key.

        ``ttl`` defaults to ``default_ttl``; ``None`` never expires and ``0`` stores nothing (and
        drops any entry with the same key). An existing key is replaced. When the cache is full
        the least recently used entry is evicted.

        Raises:
            SemanticCacheError: on invalid input, oversize input, a bad or mismatched embedding.
        """
        normalized = self._prepare_query(query, namespace)
        if not isinstance(response, str):
            raise SemanticCacheError("invalid_argument", "response must be a string")
        if len(response) > self._max_response_chars:
            message = f"response exceeds {self._max_response_chars} characters"
            raise SemanticCacheError("input_too_large", message)
        effective_ttl = self._default_ttl if isinstance(ttl, EllipsisType) else _validate_ttl(ttl)
        tag_tuple = _validate_tags(tags)
        model_id = self._current_model_id()
        key = _make_key(namespace, model_id, normalized)
        if effective_ttl == 0.0:
            with self._lock:
                existing = self._entries.get(key)
                if existing is not None:
                    self._remove(existing)
            return key
        vector = self._embed(normalized)
        events: list[dict[str, Any]] = []
        try:
            with self._lock:
                self._insert(
                    key=key,
                    namespace=namespace,
                    model_id=model_id,
                    query=normalized,
                    response=response,
                    vector=vector,
                    ttl=effective_ttl,
                    tags=tag_tuple,
                    events=events,
                )
        finally:
            self._emit(events)
        return key

    def invalidate(
        self,
        *,
        key: str | None = None,
        tag: str | None = None,
        namespace: str | None = None,
    ) -> int:
        """Remove entries by exactly one selector and return how many were removed.

        ``key`` removes one entry, ``tag`` removes every entry carrying the tag (in any
        namespace) and ``namespace`` removes a whole namespace. Invalidation is not counted as
        eviction or expiry.

        Raises:
            SemanticCacheError: ``invalid_argument`` unless exactly one selector is given.
        """
        selectors = [s for s in (key, tag, namespace) if s is not None]
        if len(selectors) != 1:
            raise SemanticCacheError("invalid_argument", "give exactly one of key, tag, namespace")
        selector = selectors[0]
        if not isinstance(selector, str) or len(selector) > _MAX_NAMESPACE_CHARS:
            raise SemanticCacheError("invalid_argument", "selector must be a short string")
        victims: list[_Entry]
        with self._lock:
            if key is not None:
                found = self._entries.get(key)
                victims = [] if found is None else [found]
            elif tag is not None:
                victims = [e for e in self._entries.values() if tag in e.tags]
            else:
                victims = [e for e in self._entries.values() if e.namespace == namespace]
            for victim in victims:
                self._remove(victim)
            return len(victims)

    def stats(self) -> CacheStats:
        """Return a snapshot of the counters and the current size."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expired=self._expired_count,
                size=len(self._entries),
            )

    def _prepare_query(self, query: str, namespace: str) -> str:
        _check_namespace(namespace)
        if not isinstance(query, str):
            raise SemanticCacheError("invalid_argument", "query must be a string")
        if len(query) > self._max_query_chars:
            message = f"query exceeds {self._max_query_chars} characters"
            raise SemanticCacheError("input_too_large", message)
        normalized = _normalize(query)
        if not normalized:
            raise SemanticCacheError("invalid_argument", "query must not be empty")
        return normalized

    def _current_model_id(self) -> str:
        model_id = self._embedder.model_id
        if not isinstance(model_id, str) or not model_id:
            raise SemanticCacheError("invalid_argument", "embedder model_id must be a string")
        if len(model_id) > _MAX_MODEL_ID_CHARS or "\x00" in model_id:
            raise SemanticCacheError("invalid_argument", "embedder model_id is not acceptable")
        return model_id

    def _embed(self, text: str) -> tuple[float, ...]:
        return _unit_vector(self._embedder.embed(text))

    def _emit(self, events: list[dict[str, Any]]) -> None:
        if self._on_event is None:
            return
        for payload in events:
            try:
                self._on_event(payload)
            except Exception:
                logger.warning("on_event callback raised; ignoring")

    @staticmethod
    def _is_expired(entry: _Entry, now: float) -> bool:
        return entry.ttl is not None and now - entry.created_at >= entry.ttl

    def _remove(self, entry: _Entry) -> None:
        del self._entries[entry.key]
        group_key = (entry.namespace, entry.model_id)
        group = self._groups[group_key]
        del group[entry.key]
        if not group:
            del self._groups[group_key]

    def _expire(self, entry: _Entry, events: list[dict[str, Any]]) -> None:
        self._remove(entry)
        self._expired_count += 1
        events.append(_event("expire", entry.namespace, None))

    @staticmethod
    def _to_hit(entry: _Entry, score: float, now: float) -> CacheHit:
        return CacheHit(
            response=entry.response,
            score=score,
            matched_query=entry.query,
            key=entry.key,
            age=max(0.0, now - entry.created_at),
        )

    def _lookup_exact(
        self, key: str, namespace: str, events: list[dict[str, Any]]
    ) -> CacheHit | None:
        now = self._clock()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._is_expired(entry, now):
            self._expire(entry, events)
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        events.append(_event("hit", namespace, 1.0))
        return self._to_hit(entry, 1.0, now)

    def _lookup_similar(
        self,
        vector: tuple[float, ...],
        namespace: str,
        model_id: str,
        events: list[dict[str, Any]],
    ) -> CacheHit | None:
        now = self._clock()
        group = self._groups.get((namespace, model_id))
        candidates = list(group.values()) if group else []
        best: _Entry | None = None
        best_score = -math.inf
        for entry in candidates:
            if self._is_expired(entry, now):
                self._expire(entry, events)
                continue
            if len(entry.vector) != len(vector):
                raise SemanticCacheError("dimension_mismatch", _DIM_MSG)
            score = _dot(entry.vector, vector)
            better = best is None or score > best_score
            tied = best is not None and score == best_score and entry.seq > best.seq
            if better or tied:
                best, best_score = entry, score
        if best is not None and best_score >= self._threshold:
            self._entries.move_to_end(best.key)
            self._hits += 1
            events.append(_event("hit", namespace, best_score))
            return self._to_hit(best, best_score, now)
        self._misses += 1
        events.append(_event("miss", namespace, best_score if best is not None else None))
        return None

    def _insert(
        self,
        *,
        key: str,
        namespace: str,
        model_id: str,
        query: str,
        response: str,
        vector: tuple[float, ...],
        ttl: float | None,
        tags: tuple[str, ...],
        events: list[dict[str, Any]],
    ) -> None:
        now = self._clock()
        group = self._groups.get((namespace, model_id))
        if group:
            for other_key, other in list(group.items()):
                if other_key == key:
                    continue
                if self._is_expired(other, now):
                    self._expire(other, events)
                    continue
                if len(other.vector) != len(vector):
                    raise SemanticCacheError("dimension_mismatch", _DIM_MSG)
                break
        existing = self._entries.get(key)
        if existing is not None:
            self._remove(existing)
        while len(self._entries) >= self._max_entries:
            oldest = next(iter(self._entries.values()))
            self._remove(oldest)
            self._evictions += 1
            events.append(_event("evict", oldest.namespace, None))
        self._seq += 1
        entry = _Entry(
            key=key,
            namespace=namespace,
            model_id=model_id,
            query=query,
            response=response,
            vector=vector,
            created_at=now,
            ttl=ttl,
            tags=tags,
            seq=self._seq,
        )
        self._entries[key] = entry
        self._groups.setdefault((namespace, model_id), {})[key] = entry
