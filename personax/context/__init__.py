from __future__ import annotations

import abc
import typing as t

from personax.resource.template import Template
from personax.types.compat.message import Messages as CompatMessages
from personax.types.context import Context
from personax.types.message import Messages
from personax.utils import AsyncContextMixin

_Primitive: t.TypeAlias = str | int | float | bool | None
_Mapping: t.TypeAlias = t.Mapping[str, _Primitive | t.Any]
BuiltT = t.TypeVar("BuiltT", bound=_Mapping)


class ContextSystem(AsyncContextMixin, abc.ABC, t.Generic[BuiltT]):
    """Abstract base class for context components in conversational AI systems.

    A ContextSystem represents a single source or processor of contextual
    information that enriches LLM prompts. Examples include knowledge retrieval
    systems (vector databases, knowledge graphs), user profile services, or
    any component that contributes domain-specific context.

    Context Building Lifecycle:
        1. Preprocess: Inspect and modify the context before building
        2. Build: Generate system-specific structured contextual data
        3. Postprocess: React to built content and modify context accordingly
        4. Parse: Convert structured data into text for LLM consumption

    Each ContextSystem must define a unique __key__ to namespace its built
    content within the shared context dictionary.

    Type Parameters:
        BuiltT: The type of structured data this system produces. Must be a
            mapping (typically TypedDict) containing the system's contextual
            information.

    Attributes:
        __key__: Unique identifier for this context system. Must be defined
            by subclasses as a class variable.

    Example:
        ```python
        class WeatherContext(t.TypedDict):
            temperature: float
            condition: str
            location: str


        class WeatherContextSystem(ContextSystem[WeatherContext]):
            __key__ = "weather"

            async def build(
                self, context: Context | str
            ) -> WeatherContext:
                # Extract location from user profile or messages
                if isinstance(context, Context):
                    location = context.context.get(
                        "profile", {}
                    ).get("location")
                else:
                    location = context  # Direct string query

                # Fetch weather data
                weather = await self.weather_api.get_current(
                    location
                )
                return WeatherContext(
                    temperature=weather.temp,
                    condition=weather.condition,
                    location=location,
                )

            async def parse(self, built: WeatherContext) -> str:
                return (
                    f"Current weather in {built['location']}: "
                    f"{built['temperature']}°C, {built['condition']}"
                )


        # Usage with ContextCompose
        weather_sys = WeatherContextSystem()
        await weather_sys.init()

        context = Context(
            messages=[
                Message(role="user", content="What should I wear?")
            ],
            context={"profile": {"location": "San Francisco"}},
        )

        # Build phase
        weather_ctx = await weather_sys.build(context)
        # Returns: {"temperature": 15.5, "condition": "Cloudy", "location": "SF"}

        # Parse phase
        text = await weather_sys.parse(weather_ctx)
        # Returns: "Current weather in San Francisco: 15.5°C, Cloudy"

        await weather_sys.close()
        ```
    """

    __key__: t.ClassVar[str]

    async def preprocess(self, context: Context) -> Context:
        """Prepare context before building system-specific content.

        The preprocess phase allows the system to:
        - Extract information from messages or existing context
        - Validate or transform data needed for the build phase
        - Enrich context for downstream systems in the composition

        This phase runs before build() and can modify the shared Context object
        that will be passed to subsequent systems in a ContextCompose.

        Args:
            context: Current conversation context with messages and accumulated
                data from preceding systems.

        Returns:
            Modified context (default returns unchanged). Changes persist for
            downstream systems.

        Example:
            ```python
            async def preprocess(self, context: Context) -> Context:
                # Extract user intent from last message
                last_msg = context.messages[-1].content
                intent = await self.intent_classifier.classify(last_msg)
                context.context["intent"] = intent
                return context
            ```
        """
        return context

    async def postprocess(self, context: Context, _built: BuiltT) -> Context:
        """React to built content and modify context after building.

        The postprocess phase allows the system to:
        - Add metadata based on what was retrieved or generated
        - Modify messages based on built content (e.g., replace images with text)
        - Enrich context for downstream systems using insights from build phase

        This phase runs after build() and can leverage the built content to
        make informed modifications to the context.

        Args:
            context: Current conversation context.
            _built: The structured data just built by this system.

        Returns:
            Modified context (default returns unchanged). Changes persist for
            downstream systems.

        Example:
            ```python
            async def postprocess(
                self, context: Context, built: MyContext
            ) -> Context:
                # Add knowledge confidence score to context
                context.context["knowledge_confidence"] = built[
                    "confidence"
                ]

                # Mark messages that were used in retrieval
                for i in built["used_message_indices"]:
                    context.messages[i].metadata[
                        "used_for_retrieval"
                    ] = True

                return context
            ```
        """
        return context

    async def parse(self, built: BuiltT) -> str | None:
        """Convert structured built data into text for LLM consumption.

        Transforms the system's structured output into natural language that
        will be incorporated into the system prompt. This text provides the
        LLM with the contextual information needed for informed responses.

        Args:
            built: The structured data produced by build().

        Returns:
            Human-readable text representation, or None to exclude this system's
            content from the final prompt.

        Example:
            ```python
            async def parse(self, built: KnowledgeContext) -> str:
                if not built["results"]:
                    return None  # No knowledge found, omit from prompt

                sections = []
                for result in built["results"]:
                    sections.append(
                        f"- {result['title']}: {result['summary']}"
                    )
                return "Relevant knowledge:\n" + "\n".join(sections)
            ```
        """

    async def init(self) -> None:
        """Initialize system resources and connections.

        Called when entering the async context manager. Should set up any
        required resources such as:
        - Database connections
        - API clients
        - Model loading
        - Cache initialization

        Example:
            ```python
            async def init(self) -> None:
                self.db_client = await DatabaseClient.connect(
                    self.db_url
                )
                self.embedding_model = await load_embedding_model()
            ```
        """

    async def close(self) -> None:
        """Clean up system resources and connections.

        Called when exiting the async context manager. Should release:
        - Database connections
        - File handles
        - Network connections
        - Cached resources

        Example:
            ```python
            async def close(self) -> None:
                await self.db_client.disconnect()
                self.embedding_model.unload()
            ```
        """

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """Validate that subclasses define a unique __key__.

        Raises:
            NotImplementedError: If __key__ is not defined.
            ValueError: If __key__ is not a non-empty string.
        """
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

    @t.overload
    async def build(self, context: Context) -> BuiltT:
        """Build from full Context object for LLM message preparation."""

    @t.overload
    async def build(self, context: str) -> BuiltT:
        """Build from string input for tool-based retrieval."""

    @abc.abstractmethod
    async def build(self, context: Context | str) -> BuiltT:
        """Generate system-specific contextual content.

        The core method where the system produces its structured contextual
        information. Supports two usage patterns:

        1. Context-based: Build from full conversation context including messages,
           user profile, and data from preceding systems in the composition.
        2. String-based: Build from direct text query, useful for tool-based
           retrieval where the system acts as a knowledge lookup function.

        Args:
            context: Either a full Context object containing conversation history
                and accumulated data, or a string query for direct retrieval.

        Returns:
            Structured data (BuiltT) containing this system's contextual
            information. This will be stored in the shared context dictionary
            under the system's __key__.

        Example:
            ```python
            async def build(
                self, context: Context | str
            ) -> KnowledgeContext:
                # String-based: direct query
                if isinstance(context, str):
                    query = context
                else:
                    # Context-based: extract query from messages
                    query = self.build_query(context)

                # Retrieve knowledge
                results = await self.vector_db.search(query, top_k=5)

                return KnowledgeContext(
                    query=query,
                    results=[
                        {
                            "title": r.title,
                            "content": r.content,
                            "score": r.score,
                        }
                        for r in results
                    ],
                    metadata={"retrieval_time": time.time()},
                )
            ```
        """


class ContextCompose(AsyncContextMixin, t.Sequence[ContextSystem[t.Any]]):
    """Orchestrator for composing multiple ContextSystems into a unified pipeline.

    ContextCompose manages the sequential execution of multiple context systems,
    coordinating their three-phase lifecycle (preprocess → build → postprocess)
    and assembling their outputs into an enriched system prompt for the LLM.

    Execution Flow:
        For each system in sequence:
        1. Preprocess: System inspects and modifies shared context
        2. Build: System generates its structured contextual data
        3. Store: Built data is stored in context dictionary under system's __key__
        4. Postprocess: System reacts to its built content

        After all systems complete:
        5. Parse: Each system's structured data is converted to text
        6. Render: Context template assembles all parsed text into system prompt
        7. Assemble: System prompt is combined with conversation messages

    Context Propagation:
        - Each system can access data from preceding systems via context.context
        - Modifications in preprocess/postprocess propagate to downstream systems
        - Built data is namespaced by __key__ to prevent conflicts

    Implements the Sequence protocol, allowing indexing and iteration over
    constituent systems.

    Attributes:
        systems: Tuple of ContextSystem instances in execution order.
        context_template: Jinja2 template for rendering the final system prompt.

    Args:
        *systems: Variable number of ContextSystem instances to compose.
            Order matters: earlier systems can influence later ones.
        context_template: Template for assembling the system prompt. Receives:
            - `systems`: Dict mapping __key__ to parsed text from each system
            - `raw`: Dict mapping __key__ to raw structured data from each system

    Example:
        ```python
        # Define template that combines all system contexts
        template = Template('''
        User Profile: {{ systems.profile }}

        Knowledge Base:
        {{ systems.knowledge }}

        Current Weather: {{ systems.weather }}

        Please respond based on the above context.
        ''')

        # Compose multiple systems
        compose = ContextCompose(
            ProfileContextSystem(...),
            KnowledgeContextSystem(...),
            WeatherContextSystem(...),
            context_template=template,
        )

        await compose.init()

        # Build enriched messages
        messages = Messages(
            messages=[
                Message(
                    role="user", content="What should I wear today?"
                )
            ]
        )
        enriched = await compose.build(
            messages,
            extras={"profile.info": {"location": "San Francisco"}},
        )

        # enriched now contains:
        # - Original user message
        # - System prompt with profile, knowledge, and weather context

        await compose.close()


        # Advanced: Access individual systems
        profile_system = compose[0]  # By index
        for system in compose:  # Iteration
            print(system.__key__)


        # Advanced: Template with raw data access
        advanced_template = Template('''
        Profile: {{ systems.profile }}

        Knowledge ({{ raw.knowledge.results|length }} results):
        {{ systems.knowledge }}

        Confidence: {{ raw.knowledge.metadata.confidence }}
        ''')
        ```

    Note:
        - Systems execute in the order provided to __init__
        - Each system's build() receives context with data from all preceding systems
        - Template has access to both parsed text (`systems`) and raw data (`raw`)
        - All systems share the same lifecycle (init/close managed together)
    """

    @t.overload
    def __getitem__(self, index: int) -> ContextSystem[t.Any]: ...

    @t.overload
    def __getitem__(self, index: slice) -> t.Sequence[ContextSystem[t.Any]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> ContextSystem[t.Any] | t.Sequence[ContextSystem[t.Any]]:
        """Access constituent systems by index or slice.

        Args:
            index: Integer index or slice for accessing systems.

        Returns:
            Single ContextSystem if indexed by int, sequence if sliced.
        """
        return self.systems[index]

    def __len__(self) -> int:
        """Get the number of constituent systems.

        Returns:
            Total number of systems in this composition.
        """
        return len(self.systems)

    def __init__(self, *systems: ContextSystem[t.Any], context_template: Template) -> None:
        self.systems = systems
        self.context_template = context_template

    async def init(self) -> None:
        """Initialize all constituent context systems.

        Calls init() on each system in sequence. Should be called before
        performing any context building operations.
        """
        for system in self.systems:
            await system.init()

    async def close(self) -> None:
        """Close and cleanup all constituent context systems.

        Calls close() on each system in sequence. Should be called when
        the composition is no longer needed.
        """
        for system in self.systems:
            await system.close()

    async def build(
        self,
        messages: Messages,
        extras: dict[str, t.Any] | None = None,
    ) -> CompatMessages:
        """Execute the complete context building pipeline.

        Orchestrates the three-phase lifecycle for all systems:
        1. Sequential Execution: For each system:
           - Preprocess: System modifies shared context
           - Build: System generates its contextual data
           - Store: Built data saved to context[system.__key__]
           - Postprocess: System reacts to built content

        2. Assembly: After all systems complete:
           - Parse: Convert each system's data to text
           - Render: Apply context template to generate system prompt
           - Combine: Merge system prompt with conversation messages

        Args:
            messages: Input conversation messages.
            extras: Additional context data to include. Typically used to provide
                initial data like user profiles or session information.

        Returns:
            CompatMessages containing the original conversation messages plus
            an enriched system prompt with all contextual information.

        Example:
            ```python
            messages = Messages(
                messages=[
                    Message(role="user", content="Recommend a book")
                ]
            )

            enriched = await compose.build(
                messages,
                extras={
                    "profile.info": {
                        "name": "Alice",
                        "interests": ["sci-fi", "philosophy"],
                    }
                },
            )

            # enriched.system_prompt now contains:
            # - User profile information
            # - Knowledge about relevant books
            # - Any other context from configured systems
            ```

        Note:
            The context dictionary accumulates data through the pipeline:
            - Starts with extras (if provided)
            - Each system adds its built data under its __key__
            - Systems can access preceding systems' data via context.context
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
        """Assemble final enriched messages with system prompt.

        Performs the final assembly phase:
        1. Parse each system's raw structured data into text
        2. Render the context template with parsed and raw data
        3. Combine the system prompt with conversation messages

        Args:
            context: Fully processed context containing all systems' built data.

        Returns:
            CompatMessages with system prompt containing all contextual
            information and the original (potentially modified) messages.

        Note:
            The template receives two dictionaries:
            - `systems`: Parsed text from each system (via parse())
            - `raw`: Raw structured data from each system (via build())

            This allows templates to use either formatted text or access
            raw data for conditional logic or detailed formatting.
        """
        raw = context.context.copy()
        parsed = {sys.__key__: await sys.parse(raw[sys.__key__]) for sys in self.systems}
        sys_prompt = self.context_template.render(systems=parsed, raw=raw)
        return CompatMessages.from_raws(
            raws=Messages.model_construct(messages=context.messages),
            sys_prompt=sys_prompt,
        )
