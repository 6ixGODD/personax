from __future__ import annotations

from personax.types import BaseSchema


class Usage(BaseSchema):
    """Token usage statistics for LLM completion.

    Tracks the number of tokens consumed during a completion request,
    broken down by prompt (input) and completion (output) tokens.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated in the completion.
        total_tokens: Sum of prompt_tokens and completion_tokens.

    Example:
        ```python
        completion = await system.complete(messages, model="gpt-4")

        if completion.usage:
            print(f"Prompt: {completion.usage.prompt_tokens} tokens")
            print(f"Completion: {completion.usage.completion_tokens} tokens")
            print(f"Total: {completion.usage.total_tokens} tokens")

            # Calculate cost (example rates)
            prompt_cost = completion.usage.prompt_tokens * 0.00003
            completion_cost = completion.usage.completion_tokens * 0.00006
            total_cost = prompt_cost + completion_cost
            print(f"Estimated cost: ${total_cost:.4f}")
        ```
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    __slots__ = (
        "completion_tokens",
        "prompt_tokens",
        "total_tokens",
    )

    def __init__(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
