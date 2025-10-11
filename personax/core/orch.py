from __future__ import annotations

import typing as t

from persomna.core import Persomna
from persomna.types import BaseModel


class Orch:
    _registry: t.ClassVar[t.Dict[str, Persomna]] = {}

    @classmethod
    def from_config(cls, config: BaseModel) -> t.Self:
        pass

    def fetch(self, id: str) -> Persomna:
        return self._registry[id]
