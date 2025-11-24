from __future__ import annotations

import typing as t


class Usage(t.NamedTuple):
    """Token usage statistics for LLM completion.

    Tracks the number of tokens consumed during a completion request,
    broken down by prompt (input) and completion (output) tokens.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated in the completion.
        total_tokens: Sum of prompt_tokens and completion_tokens.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
