from __future__ import annotations

import types
import typing as t


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
        return False


@t.runtime_checkable
class ContextMixin(t.Protocol):

    def init(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> t.Self:
        self.init()
        return self

    def __exit__(
        self,
        exc_type: t.Type[BaseException] | None,
        exc_val: BaseException | None,
        traceback: types.TracebackType | None,
        /,
    ) -> t.Literal[False]:
        self.close()
        if exc_type is not None:
            raise (exc_val or exc_type()).with_traceback(traceback) from exc_val
        return False
