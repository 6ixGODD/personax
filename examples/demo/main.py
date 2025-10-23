"""
PersonaX Framework Demo - Interactive AI Assistant
===================================================
This demo showcases the PersonaX framework's key features:
- Context-aware conversations with user profiling
- Real-time weather information via tool integration
- Seamless async conversation flow
- Streaming and non-streaming modes
"""

from __future__ import annotations

import asyncio

import halo
from personax.completion.openai import OpenAICompletion
from personax.completion.openai import OpenAIConfig
from personax.context import ContextCompose
from personax.context.profile import Info
from personax.context.profile import ProfileContextSystem
from personax.core import Core
from personax.core import PersonaX
from personax.resources.rest.ip.baidu import BaiduIpLocationService
from personax.resources.rest.weather.amap import AmapWeatherInfoService
from personax.resources.template import WatchedJ2Template
from personax.tools.weather import GetWeather
from personax.types.message import Message
from personax.types.message import Messages
import yaml

# ============================================================================
# Configuration
# ============================================================================

# Load config
with open(".config.yml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def get_user_info() -> Info:
    """Provide user information for context-aware responses."""
    user = CONFIG["user"]
    return Info(
        prefname=user["prefname"],
        ip=user["ip"],
        user_agent=user["user_agent"],
        platform=user["platform"],
        timezone="Asia/Shanghai",
    )


def setup_persona() -> PersonaX:
    """Set up the PersonaX assistant with all necessary components."""
    # LLM completion service
    llm = CONFIG["llm"]
    completion = OpenAICompletion(
        openai_config=OpenAIConfig(
            base_url=llm["base_url"],
            api_key=llm["api_key"],
            model=llm["model"],
        )
    )

    # Context management with user profiling
    profile_system = ProfileContextSystem(
        ip_service=BaiduIpLocationService(ak=CONFIG["services"]["baidu_ip"]["ak"]),
        template=WatchedJ2Template(CONFIG["templates"]["profile"]),
    )
    context = ContextCompose(
        profile_system, context_template=WatchedJ2Template(CONFIG["templates"]["main"])
    )

    # Weather tool
    weather_tool = GetWeather(
        weather_srv=AmapWeatherInfoService(key=CONFIG["services"]["amap_weather"]["key"])
    )

    # Create core and persona
    persona_cfg = CONFIG["persona"]

    class WeatherAssistant(PersonaX):
        name = persona_cfg["name"]
        version = persona_cfg["version"]
        scenario = persona_cfg["scenario"]

    core = Core(
        completion=completion,
        context=context,
        toolset=[weather_tool],
        model_id=WeatherAssistant.id,
    )

    return WeatherAssistant(core)


# ============================================================================
# Interactive Chat
# ============================================================================


async def chat():
    """Main chat loop."""
    persona = setup_persona()
    history: list[Message] = []
    streaming = CONFIG["chat"].get("streaming_mode", True)
    show_spinner = CONFIG["chat"].get("show_spinner", True)
    spinner = halo.Halo(text="Thinking", spinner="dots")

    # Welcome message
    print("\n" + "=" * 60)
    print("🤖 Welcome to PersonaX Interactive Demo!")
    print("=" * 60)
    print(f"Persona ID: {persona.id}")
    print(f"Mode: {'Streaming ⚡' if streaming else 'Standard 📝'}")
    print("\nI'm your AI assistant powered by PersonaX framework.")
    print("I can help you with weather information and casual chat!")
    print("\nCommands:")
    print("  • Type 'exit' or 'quit' to end the conversation")
    print("  • Type 'clear' to reset conversation history")
    print("  • Type 'stream' to toggle streaming mode")
    print("=" * 60 + "\n")

    async with persona:
        while True:
            # Get user input
            try:
                user_input = input("💬 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("exit", "quit"):
                print("\n👋 Thanks for trying PersonaX! Goodbye!\n")
                break
            if user_input.lower() == "clear":
                history.clear()
                print("\n🔄 Conversation history cleared!\n")
                continue
            if user_input.lower() == "stream":
                streaming = not streaming
                print(f"\n🔄 Switched to {'Streaming ⚡' if streaming else 'Standard 📝'} mode!\n")
                continue

            # Process message
            history.append(Message(role="user", content=user_input))

            if streaming:
                # Streaming mode
                print("🤖 Assistant: ", end="", flush=True)
                assistant_message = ""
                response = await persona.complete(
                    messages=Messages(messages=history),
                    stream=True,
                    extras={"profile.info": get_user_info()},
                )
                async for chunk in response:
                    print(chunk.delta.content or "", end="", flush=True)
                    assistant_message += chunk.delta.content or ""
                print("\n")
            else:
                # Standard mode
                if show_spinner:
                    spinner.start()
                response = await persona.complete(
                    messages=Messages(messages=history),
                    stream=False,
                    extras={"profile.info": get_user_info()},
                )
                assistant_message = response.message.content
                if show_spinner:
                    spinner.stop()
                print(f"🤖 Assistant: {assistant_message}\n")

            history.append(Message(role="assistant", content=assistant_message))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    asyncio.run(chat())
