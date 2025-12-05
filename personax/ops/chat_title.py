from __future__ import annotations

from personax.completion import CompletionSystem
from personax.resource.template import Template
from personax.types.message import Message
from personax.types.message import Messages


class ChatTitle:
    def __init__(self, completion: CompletionSystem, prompt: Template):
        self.completion = completion
        self.prompt = prompt

    async def __call__(self, messages: list[Message]) -> str:
        prompt_filled = self.prompt.render(
            messages="\n".join(f"{message.role}: {message.content}" for message in messages)
        )
        response = await self.completion.complete(
            messages=Messages(messages=[Message(role="user", content=prompt_filled)]),
            model="chat-title",
        )
        return response.message.content.strip().strip('"')
