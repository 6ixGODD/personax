from __future__ import annotations

import typing as t


@t.runtime_checkable
class Storage(t.Protocol):

    async def upload(self, data: bytes | t.IO[bytes], key: str) -> str:
        pass

    async def download(self, key: str, stream: bool = False) -> bytes | t.AsyncIterable[bytes]:
        pass

    async def delete(self, key: str) -> None:
        pass

    async def list(self, prefix: str = '') -> t.List[str]:
        pass
