from __future__ import annotations

import typing as t

from personax.core import PersonaX
from personax.types import BaseModel


class Orch:
    _registry: t.ClassVar[t.Dict[str, PersonaX]] = {}

    @classmethod
    def from_config(cls, config: BaseModel) -> t.Self:
        pass

    def fetch(self, id_: str, /) -> PersonaX:
        return self._registry[id_]
