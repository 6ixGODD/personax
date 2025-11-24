from __future__ import annotations

from personax.completion import CompletionSystem
from personax.resource.template import Template
from personax.types.message import Message


class ChatTitle:
    def __init__(self, completion: CompletionSystem, prompt: Template):
        self.completion = completion
        self.prompt = prompt

    async def gen(self, context: Message): ...
