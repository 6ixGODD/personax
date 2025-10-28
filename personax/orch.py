from __future__ import annotations

import builtins
import typing as t

from personax import PersonaX

T = t.TypeVar("T", bound=PersonaX)

class Orch(t.MutableMapping[str, T]):
    def __init__(self) -> None:
        self._ins: dict[str, T] = {}

    def register(self, persona: T) -> None:
        if persona.id in self._ins:
            raise ValueError(f"Persona with id '{persona.id}' is already registered.")
        self._ins[persona.id] = persona

    def unregister(self, persona_id: str) -> None:
        if persona_id not in self._ins:
            raise KeyError(f"Persona with id '{persona_id}' is not registered.")
        del self._ins[persona_id]

    def get(self, persona_id: str, /) -> T:
        if persona_id not in self._ins:
            raise KeyError(f"Persona with id '{persona_id}' is not registered.")
        return self._ins[persona_id]

    def list(self) -> builtins.list[T]:
        return list(self._ins.values())

    def keys(self) -> t.KeysView[str]:
        return self._ins.keys()

    def __getitem__(self, key: str, /) -> T:
        return self.get(key)

    def __setitem__(self, key: str, /, value: T) -> None:
        if key != value.id:
            raise ValueError("Key must be the same as PersonaX.id")
        self.register(value)

    def __delitem__(self, key: str, /) -> None:
        self.unregister(key)

    def __iter__(self) -> t.Iterator[str]:
        return iter(self._ins)

    def __len__(self) -> int:
        return len(self._ins)
