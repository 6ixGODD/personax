from __future__ import annotations

import abc
import typing as t

from personax.helpers.mixin import AsyncContextMixin
from personax.tools import BaseTool
from personax.types.completion import Completion
from personax.types.completion_chunk import CompletionChunk
from personax.types.message import Messages
from personax.types.stream import AsyncStream
from personax.utils import UNSET
from personax.utils import Unset


class CompletionSystem(AsyncContextMixin, abc.ABC):

    @abc.abstractmethod
    async def complete(self,
                       messages: Messages,
                       *,
                       tools: t.Sequence[BaseTool] = (),
                       chatcmpl_id: str | Unset = UNSET,
                       stream: bool = False,
                       max_completion_tokens: int | Unset = UNSET,
                       model: str,
                       _prompt_cache_key: str | Unset = UNSET,
                       **kwargs: t.Any) -> Completion | AsyncStream[CompletionChunk]:
        ...
