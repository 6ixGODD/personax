from __future__ import annotations

import time
import typing as t

from cachetools import LRUCache


class CachetoolsKVCache:
    """A KVCache implementation based on cachetools.LRUCache.

    Note:
        This uses LRUCache for simplicity. For TTL support, we manually
        track expiry times since cachetools.TTLCache has global TTL limitations.

    Args:
        maxsize: Maximum number of items in the cache
        default_ttl: Default TTL in seconds (None means no expiration)
    """

    def __init__(self, maxsize: int = 1024, default_ttl: int | None = None):
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._cache = LRUCache(maxsize=maxsize)  # type: LRUCache[str, t.Any]
        self._expiry = {}  # type: t.Dict[str, float | None]

    def _is_expired(self, key: str) -> bool:
        expiry = self._expiry.get(key)
        return expiry is not None and time.time() > expiry

    def _cleanup_expired_key(self, key: str) -> bool:
        if self._is_expired(key):
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def __getitem__(self, key: str) -> t.Any:
        if self._cleanup_expired_key(key):
            raise KeyError(key)
        return self._cache[key]

    def __setitem__(self, key: str, value: t.Any) -> None:
        self._cache[key] = value
        if self._default_ttl is not None:
            self._expiry[key] = time.time() + self._default_ttl
        else:
            self._expiry[key] = None

    def __delitem__(self, key: str) -> None:
        if self._cleanup_expired_key(key):
            raise KeyError(key)
        del self._cache[key]
        self._expiry.pop(key, None)

    def __iter__(self) -> t.Iterator[str]:
        # Clean up expired keys
        expired = [k for k in list(self._cache.keys()) if self._is_expired(k)]
        for k in expired:
            self._cache.pop(k, None)
            self._expiry.pop(k, None)
        return iter(self._cache)

    def __len__(self) -> int:
        # Clean up expired keys
        expired = [k for k in list(self._cache.keys()) if self._is_expired(k)]
        for k in expired:
            self._cache.pop(k, None)
            self._expiry.pop(k, None)
        return len(self._cache)

    def __contains__(self, key: object, /) -> bool:
        if not isinstance(key, str):
            return False
        if self._cleanup_expired_key(key):
            return False
        return key in self._cache

    def get(self, key: str, /) -> t.Any:
        try:
            return self[key]
        except KeyError:
            return None

    def clear(self) -> None:
        self._cache.clear()
        self._expiry.clear()

    def pop(self, key: str, /) -> t.Any:
        if self._cleanup_expired_key(key):
            raise KeyError(key)
        self._expiry.pop(key, None)
        return self._cache.pop(key)

    def popitem(self) -> t.Tuple[str, t.Any]:
        if not self._cache:
            raise KeyError("popitem(): cache is empty")
        # cachetools LRUCache.popitem() removes least recently used
        key, value = self._cache.popitem()
        self._expiry.pop(key, None)
        return key, value

    def set(self, key: str, value: t.Any, /) -> None:
        self[key] = value

    def setx(self, key: str, value: t.Any, /, ttl: int | None = None) -> None:
        self._cache[key] = value
        if ttl is not None:
            self._expiry[key] = time.time() + ttl
        else:
            self._expiry[key] = None

    def ttl(self, key: str, /) -> int | None:
        if self._cleanup_expired_key(key):
            raise KeyError(key)
        if key not in self._cache:
            raise KeyError(key)

        expiry = self._expiry.get(key)
        if expiry is None:
            return None
        remaining = int(expiry - time.time())
        return max(0, remaining)

    def expire(self, key: str, /, ttl: int | None = None) -> None:
        if self._cleanup_expired_key(key):
            raise KeyError(key)
        if key not in self._cache:
            raise KeyError(key)

        if ttl is not None:
            self._expiry[key] = time.time() + ttl
        else:
            self._expiry[key] = None

    def incr(self, key: str, /, amount: int = 1) -> int:
        if key not in self._cache or self._cleanup_expired_key(key):
            self[key] = 0

        value = self._cache[key]
        if not isinstance(value, int):
            raise TypeError(f"value is not an integer: {type(value).__name__}")

        new_value = value + amount
        self._cache[key] = new_value
        return new_value

    def decr(self, key: str, /, amount: int = 1) -> int:
        return self.incr(key, -amount)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(maxsize={self._maxsize}, items={len(self)})"
