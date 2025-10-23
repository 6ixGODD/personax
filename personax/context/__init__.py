from __future__ import annotations

import abc
import typing as t

from personax.resources.template import Template
from personax.types.compat.message import Messages as CompatMessages
from personax.types.context import Context
from personax.types.message import Messages
from personax.utils import AsyncContextMixin

_Primitive: t.TypeAlias = str | int | float | bool | None
_Mapping: t.TypeAlias = t.Mapping[str, _Primitive | t.Any]
BuiltT = t.TypeVar("BuiltT", bound=_Mapping)


class ContextSystem(abc.ABC, t.Generic[BuiltT]):
    """Abstract base class for context systems that process and enrich
    conversational context.

    A ContextSystem represents a single source or processor of contextual
    information (e.g., vector database, knowledge graph, user profile service)
    that can:
    1. Preprocess the context before building
    2. Build its specific contextual content
    3. Postprocess the context after building
    4. Parse its built content into a string format for LLM consumption

    Type Parameters:
        BuiltT: The type of structured data this system produces (must be a
            mapping)
    """

    __key__: t.ClassVar[str]  # Unique key identifying this context system

    async def preprocess(self, context: Context) -> Context:
        """Preprocess the context before this system builds its content.

        This phase allows the system to:
        - Inspect existing messages and context
        - Extract information needed for the build phase
        - Modify the context for downstream systems

        Args:
            context: The current context containing messages and accumulated
                data

        Returns:
            Modified context (default implementation returns unchanged)
        """
        return context  # override if needed

    async def postprocess(self, context: Context, _built: BuiltT) -> Context:
        """Postprocess the context after this system has built its content.

        This phase allows the system to:
        - Add metadata based on what was built
        - Modify the context for downstream systems
        - Perform cleanup or validation

        Args:
            context: The current context
            _built: The content this system just built

        Returns:
            Modified context (default implementation returns unchanged)
        """
        return context  # override if needed

    async def parse(self, built: BuiltT) -> str | None:
        """Parse the built structured data into a string format for LLM
        consumption.

        This method converts the system's structured output into human-readable
        text that will be included in the system prompt.

        Args:
            built: The structured data built by this system

        Returns:
            String representation for the LLM, or None to omit from final prompt
        """

    async def init(self) -> None:
        """Initialize the context system.

        This method should set up any required resources, connections, or state.
        Called when entering the async context manager.
        """

    async def close(self) -> None:
        """Clean up and close the context system.

        This method should release resources and perform cleanup. Called when
        exiting the async context manager.
        """

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "__key__"):
            raise NotImplementedError(
                f"{cls.__name__} must define a unique __key__ class variable."
            )
        if not isinstance(cls.__key__, str) or not cls.__key__:
            raise ValueError(f"{cls.__name__}.__key__ must be a non-empty string.")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__key__})"

    __repr__ = __str__

    # For preparing messages for LLM calls.
    @t.overload
    async def build(self, context: Context) -> BuiltT:
        """Build from a full Context object (for preparing messages for LLM
        calls)."""

    # For tool calls with string input/output.
    @t.overload
    async def build(self, context: str) -> BuiltT:
        """Build from a string input (for tool calls with string
        input/output)."""

    @abc.abstractmethod
    async def build(self, context: Context | str) -> BuiltT:
        """Build the contextual content for this system.

        This is the core method where the system generates its specific
        contextual information based on the input.

        Args:
            context: Either a full Context object or a string input

        Returns:
            Structured data (BuiltT) containing this system's contextual
            information
        """


class ContextCompose(t.Sequence[ContextSystem[t.Any]], AsyncContextMixin):
    """Composition of multiple ContextSystems into a unified context pipeline.

    ContextCompose orchestrates multiple context systems, running them in
    sequence through a three-phase process:
    1. Preprocess: Each system can inspect and modify the context
    2. Build: Each system generates its contextual content
    3. Postprocess: Each system can react to what was built

    The final output assembles all system contexts into a structured message
    format with an enriched system prompt.

    Implements Sequence protocol to allow indexing and iteration over systems.

    Args:
        *systems: Variable number of ContextSystem instances to compose
        context_template: Jinja2 template string for assembling the final
            system prompt. Receives 'systems' (parsed content) and 'raw' (raw
            built content) as template variables.

    """

    @t.overload
    def __getitem__(self, index: int) -> ContextSystem[t.Any]: ...

    @t.overload
    def __getitem__(self, index: slice) -> t.Sequence[ContextSystem[t.Any]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ContextSystem[t.Any] | t.Sequence[ContextSystem[t.Any]]:
        return self.systems[index]

    def __len__(self) -> int:
        return len(self.systems)

    def __init__(self, *systems: ContextSystem[t.Any], context_template: Template) -> None:
        self.systems = systems
        self.context_template = context_template

    async def init(self) -> None:
        for system in self.systems:
            await system.init()

    async def close(self) -> None:
        for system in self.systems:
            await system.close()

    async def build(
        self,
        messages: Messages,
        extras: dict[str, t.Any] | None = None,
    ) -> CompatMessages:
        """Build the complete contextual message structure by running all
        systems.

        This method orchestrates the three-phase process:
        1. For each system, run preprocess → build → postprocess
        2. Accumulate all built content in the context
        3. Parse and assemble everything into the final message format

        Args:
            messages: The input messages from the conversation
            extras: Additional context data to include

        Returns:
            CompatMessages with enriched system prompt containing all context
        """
        messages_ = messages.model_copy()
        context = Context(messages=list(messages_.messages), context=extras or {})

        for system in self.systems:
            # 1. Preprocess phase: allow each system to modify the context
            context = await system.preprocess(context)
            # 2. Build phase: allow each system to add its content
            content = await system.build(context)
            context.context[system.__key__] = content
            # 3. Postprocess phase: allow each system to modify the context again
            context = await system.postprocess(context, content)

        # Final assembly phase
        return await self.build_final(context)

    async def build_final(self, context: Context) -> CompatMessages:
        """Assemble the final message structure with enriched system prompt.

        This method:
        1. Parses each system's raw built content into string format
        2. Renders the context template with all parsed content
        3. Combines the rendered system prompt with the original messages

        Args:
            context: The fully processed context containing all systems' built
                content

        Returns:
            CompatMessages with system prompt containing all contextual
            information
        """
        raw = context.context.copy()
        parsed = {sys.__key__: await sys.parse(raw[sys.__key__]) for sys in self.systems}
        sys_prompt = self.context_template.render(systems=parsed, raw=raw)
        return CompatMessages.from_raws(
            raws=Messages.model_construct(messages=context.messages),
            sys_prompt=sys_prompt,
        )
