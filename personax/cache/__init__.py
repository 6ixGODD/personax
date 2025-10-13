from __future__ import annotations

import typing as t

from personax import __title__


# pylint: disable=too-few-public-methods
class KeyBuilder:
    """Utility class for building cache keys with a consistent format.

    Attributes:
        split_char (str): Character used to split parts of the key.
        prefix (str): Prefix to prepend to all keys.
    """

    def __init__(self, split_char: str = ':', prefix: str = __title__) -> None:
        self.split_char = split_char
        self.prefix = prefix

    def build(self, *parts: str) -> str:
        """Build a cache key by joining the prefix and parts with the split
        character.

        Args:
            *parts (str): Parts to include in the key.
            
        Returns:
            str: The constructed cache key.
        """
        return self.split_char.join((self.prefix, *parts))


class KVCache(t.Protocol):
    """Protocol for a key-value cache with TTL support.

    This protocol defines a mapping-like interface with cache-specific operations.

    Exception behavior:
    - __getitem__, __delitem__, pop (without default) should raise KeyError when key doesn't exist
    - get returns None when key doesn't exist
    - incr/decr initialize non-existent keys to 0
    - ttl and expire raise KeyError when key doesn't exist
    """

    def __getitem__(self, key: str) -> t.Any:
        """Retrieve an item from the cache by key.

        Raises:
            KeyError: If the key does not exist or has expired
        """

    def __setitem__(self, key: str, value: t.Any) -> None:
        """Set an item in the cache with the specified key and value."""

    def __delitem__(self, key: str) -> None:
        """Delete an item from the cache by key.

        Raises:
            KeyError: If the key does not exist
        """

    def __iter__(self) -> t.Iterator[str]:
        """Return an iterator over the keys in the cache."""

    def __len__(self) -> int:
        """Return the number of items in the cache."""

    def __contains__(self, key: object, /) -> bool:
        """Check if the cache contains a specific key."""

    def get(self, key: str, /) -> t.Any:
        """Get an item from the cache, returning None if the key does not
        exist."""

    def clear(self) -> None:
        """Clear all items from the cache."""

    def pop(self, key: str, /) -> t.Any:
        """Remove and return an item from the cache by key.

        Raises:
            KeyError: If the key does not exist
        """

    def popitem(self) -> t.Tuple[str, t.Any]:
        """Remove and return an arbitrary (key, value) pair from the cache.

        Raises:
            KeyError: If the cache is empty
        """

    def set(self, key: str, value: t.Any, /) -> None:
        """Set an item in the cache with the specified key and value."""

    def setx(self, key: str, value: t.Any, /, ttl: int | None = None) -> None:
        """Set an item in the cache with the specified key, value, and
        optional time-to-live (TTL)."""

    def ttl(self, key: str, /) -> int | None:
        """Get the time-to-live (TTL) for a specific key in the cache.

        Returns:
            Remaining TTL in seconds, or None if no expiration

        Raises:
            KeyError: If the key does not exist
        """

    def expire(self, key: str, /, ttl: int | None = None) -> None:
        """Set the time-to-live (TTL) for a specific key in the cache.

        Raises:
            KeyError: If the key does not exist
        """

    def incr(self, key: str, /, amount: int = 1) -> int:
        """Increment the integer value of a key by the given amount. If the key
        does not exist, it is set to 0 before performing the operation.

        Raises:
            TypeError: If the value is not an integer
        """

    def decr(self, key: str, /, amount: int = 1) -> int:
        """Decrement the integer value of a key by the given amount. If the key
        does not exist, it is set to 0 before performing the operation.

        Raises:
            TypeError: If the value is not an integer
        """
