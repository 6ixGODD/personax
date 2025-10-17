from __future__ import annotations

import abc
import typing as t

from personax.tools import BaseToolType
from personax.types.compat.message import Messages
from personax.types.completion import Completion
from personax.types.completion_chunk import CompletionChunk
from personax.types.stream import AsyncStream
from personax.utils import AsyncContextMixin
from personax.utils import UNSET
from personax.utils import Unset


class CompletionSystem(AsyncContextMixin, abc.ABC):

    @abc.abstractmethod
    async def complete(
        self,
        messages: Messages,
        *,
        tools: t.Sequence[BaseToolType] = (),
        chatcmpl_id: str | Unset = UNSET,
        stream: bool = False,
        max_completion_tokens: int | Unset = UNSET,
        model: str,
        _prompt_cache_key: str | Unset = UNSET,
        **kwargs: t.Any
    ) -> Completion | AsyncStream[CompletionChunk]:
        ...
