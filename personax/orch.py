from __future__ import annotations

import builtins
import typing as t

from personax import PersonaX

T = t.TypeVar("T", bound=PersonaX)


class Orch(t.MutableMapping[str, T]):
    """Orchestrator for managing multiple PersonaX instances.

    Provides a registry for PersonaX personas, enabling dynamic lookup and
    management of multiple model configurations. Implements the MutableMapping
    interface for dict-like access by persona ID.

    The orchestrator allows:
    - Registration of multiple personas with unique IDs
    - Dynamic persona lookup by ID
    - Iteration over registered personas
    - Management of persona lifecycle

    Type Parameters:
        T: PersonaX or subclass type managed by this orchestrator.

    Example:
        ```python
        # Create orchestrator
        orch = Orch[PersonaX]()

        # Register personas
        assistant_general = Assistant(core_general)
        assistant_s2s = AssistantS2S(core_s2s)

        orch.register(assistant_general)  # ID: "assistant@general"
        orch.register(assistant_s2s)  # ID: "assistant@s2s"


        # Access by ID
        persona = orch.get("assistant@s2s")
        persona = orch["assistant@s2s"]  # Dict-like access


        # List all personas
        for persona in orch.list():
            print(persona.id)


        # Iterate over IDs
        for persona_id in orch:
            print(persona_id)


        # Dict-like operations
        orch["assistant@medical"] = assistant_medical  # Register
        del orch["assistant@general"]  # Unregister
        print(len(orch))  # Count personas


        # Practical usage
        async def handle_request(scenario: str, messages: Messages):
            persona_id = f"assistant@{scenario}"
            if persona_id in orch:
                persona = orch[persona_id]
                return await persona.complete(messages)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
        ```

    Note:
        - Persona IDs must be unique within an orchestrator
        - Attempting to register duplicate IDs raises ValueError
        - Dict-style setitem requires key to match persona.id
        - Personas are not automatically initialized or closed
    """

    def __init__(self) -> None:
        self._ins: dict[str, T] = {}

    def register(self, persona: T) -> None:
        """Register a persona with the orchestrator.

        Args:
            persona: PersonaX instance to register.

        Raises:
            ValueError: If a persona with the same ID is already registered.

        Example:
            ```python
            orch = Orch()
            orch.register(Assistant(core))
            orch.register(AssistantS2S(core))
            ```
        """
        if persona.id in self._ins:
            raise ValueError(f"Persona with id '{persona.id}' is already registered.")
        self._ins[persona.id] = persona

    def unregister(self, persona_id: str) -> None:
        """Unregister a persona from the orchestrator.

        Args:
            persona_id: Unique ID of the persona to unregister.

        Raises:
            KeyError: If no persona with the given ID is registered.

        Example:
            ```python
            orch.unregister("assistant@s2s")
            ```
        """
        if persona_id not in self._ins:
            raise KeyError(f"Persona with id '{persona_id}' is not registered.")
        del self._ins[persona_id]

    def get(self, persona_id: str, /) -> T:
        """Retrieve a persona by ID.

        Args:
            persona_id: Unique ID of the persona.

        Returns:
            The registered PersonaX instance.

        Raises:
            KeyError: If no persona with the given ID is registered.

        Example:
            ```python
            persona = orch.get("assistant@medical")
            completion = await persona.complete(messages)
            ```
        """
        if persona_id not in self._ins:
            raise KeyError(f"Persona with id '{persona_id}' is not registered.")
        return self._ins[persona_id]

    def list(self) -> builtins.list[T]:
        """Get a list of all registered personas.

        Returns:
            List of all PersonaX instances in the orchestrator.

        Example:
            ```python
            for persona in orch.list():
                print(
                    f"{persona.id}: {persona.name} v{persona.version}"
                )
            ```
        """
        return list(self._ins.values())

    def keys(self) -> t.KeysView[str]:
        """Get a view of all registered persona IDs.

        Returns:
            View of persona ID strings.
        """
        return self._ins.keys()

    def __getitem__(self, key: str, /) -> T:
        """Dict-like access to retrieve persona by ID.

        Args:
            key: Persona ID.

        Returns:
            The registered PersonaX instance.

        Raises:
            KeyError: If no persona with the given ID is registered.
        """
        return self.get(key)

    def __setitem__(self, key: str, /, value: T) -> None:
        """Dict-like registration of a persona.

        Args:
            key: Must match value.id.
            value: PersonaX instance to register.

        Raises:
            ValueError: If key doesn't match value.id or ID is already registered.

        Example:
            ```python
            orch["assistant@s2s"] = assistant_s2s
            ```
        """
        if key != value.id:
            raise ValueError("Key must be the same as PersonaX.id")
        self.register(value)

    def __delitem__(self, key: str, /) -> None:
        """Dict-like unregistration of a persona.

        Args:
            key: Persona ID to unregister.

        Raises:
            KeyError: If no persona with the given ID is registered.
        """
        self.unregister(key)

    def __iter__(self) -> t.Iterator[str]:
        """Iterate over persona IDs.

        Returns:
            Iterator of persona ID strings.
        """
        return iter(self._ins)

    def __len__(self) -> int:
        """Get the number of registered personas.

        Returns:
            Count of registered personas.
        """
        return len(self._ins)
