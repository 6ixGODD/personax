from __future__ import annotations

import functools as ft
import inspect
import threading
import types
import typing as t

_C = t.TypeVar("_C")


def singleton(cls: t.Type[_C]) -> t.Callable[..., _C]:
    """Thread-safe singleton decorator.

    This decorator ensures that only one instance of the decorated class
    exists across the program. The instance is created the first time the
    class is instantiated and the same instance is returned for all
    subsequent instantiations. Thread safety is maintained using a reentrant
    lock to protect access to the internal instance dictionary.

    This pattern is suitable when you want to add singleton behavior to
    selected classes without modifying their inheritance hierarchy or metaclass.

    Args:
        cls: The class to be decorated as a singleton.

    Returns:
        A wrapper function that returns the singleton instance of the class.
    """
    # Dictionary to keep singleton instances for each decorated class.
    instances: t.Dict[t.Type[_C], t.Any] = {}
    lock = threading.RLock()

    @ft.wraps(cls)
    def get_instance(*args: t.Any, **kwargs: t.Any) -> t.Any:
        nonlocal instances
        # If instance for this class does not exist, create it with thread
        # safety.
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        # Return the singleton instance.
        return instances[cls]

    return get_instance


# pylint: disable=invalid-name
class classproperty(property):

    def __get__(self, __instance: t.Any, __owner: type | None = None) -> t.Any:
        if not callable(self.fget):
            raise TypeError("fget must be callable")
        return self.fget(__owner)


@ft.lru_cache(maxsize=None)
def _get_func_params(fn: t.Callable[..., t.Any]) -> t.Set[str]:
    return set(inspect.signature(fn).parameters.keys())


def filter_kwargs(fn: t.Callable[..., t.Any],
                  kwargs: t.Dict[str, t.Any],
                  pref: str = "") -> t.Dict[str, t.Any]:
    """
    Filter out invalid keyword arguments for a given function by comparing
    the provided keyword arguments to the function's signature. Only valid
    keyword arguments are returned.

    Args:
        fn: The function to filter keyword arguments for.
        kwargs: The keyword arguments to filter.
        pref: The prefix to remove from keyword argument names before
        checking. Defaults to "".

    Returns:
        The filtered keyword arguments with valid parameter names only.
    """
    valid_params = _get_func_params(fn)

    if pref:
        # Remove prefix and filter
        filtered = {}
        for key, value in kwargs.items():
            if key.startswith(pref):
                param_name = key[len(pref):]
                if param_name in valid_params and param_name not in {"self", "cls"}:
                    filtered[param_name] = value
        return filtered
    # Direct filtering without prefix removal
    return {
        key: value
        for key, value in kwargs.items()
        if key in valid_params and key not in {"self", "cls"}
    }


class Unset:

    def __repr__(self) -> str:
        return "<UNSET>"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unset)

    def __hash__(self) -> int:
        return hash("UNSET")


UNSET = Unset()


def flatten_dict(
    _dict: t.Mapping[str, t.Any],
    /,
    sep: str = ".",
    _parent: str = "",
) -> t.Dict[str, t.Any]:
    items = []  # type: t.List[tuple[str, t.Any]]
    for k, v in _dict.items():
        key = f"{_parent}{sep}{k}" if _parent else k
        if isinstance(v, t.Mapping):
            items.extend(flatten_dict(v, _parent=key, sep=sep).items())
        else:
            items.append((key, v))
    return dict(items)


@t.runtime_checkable
class AsyncContextMixin(t.Protocol):

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> t.Self:
        await self.init()
        return self

    async def __aexit__(
        self,
        exc_type: t.Type[BaseException] | None,
        exc_val: BaseException | None,
        traceback: types.TracebackType | None,
        /,
    ) -> t.Literal[False]:
        await self.close()
        if exc_type is not None:
            raise (exc_val or exc_type()).with_traceback(traceback) from exc_val
        return False  # do not suppress exceptions
