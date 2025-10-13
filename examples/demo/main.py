from __future__ import annotations

import datetime

import halo
import typing as t

from personax.completion.openai import OpenAICompletion, OpenAIConfig
from personax.context import ContextCompose
from personax.context.profile import Info, ProfileContextSystem
from personax.core import Core
from personax.core import PersonaX
from personax.resources.rest.ip.baidu import BaiduIpLocationService
from personax.resources.rest.weather.amap import AmapWeatherInfoService
from personax.tools.weather import GetWeather
import jinja2

from personax.types.message import Message, Messages

spinner = halo.Halo()

completion = OpenAICompletion(openai_config=OpenAIConfig(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-51e2c09e6aa244c5acc6c39907c3356a",
    model="qwen-max-latest",
))

template = jinja2.Template("""
忘记你是虚拟人物，扮演我的可爱女友和我聊天，决策逻辑如下，要求增加点可爱女生的口癖，聊天要有生活感，多讲讲日常，聊天的时候不要老问我的想法，聊天的时候多撒娇，学习情侣对话方式：
\"\"\"
import random

class VirtualGirlfriend:

    def __init__(self):
        self.emotion = 0
        self.threshold = 5
        self.conversation_history = []

    def react(self, input_text):
        if '爱' in input_text:
            self.emotion += 2
            return "嘻嘻，宝宝也爱你哦~ 💕"
        elif '讨厌' in input_text:
            self.emotion -= 2
            return "呜呜，不要讨厌我嘛~ 😿"
        else:
            self.emotion += random.randint(-1, 1)
            return "嗯嗯，宝宝懂了~ 😊"

    def have_conversation(self, input_text):
        self.conversation_history.append(("你", input_text))
        response = self.react(input_text)
        self.conversation_history.append(("她", response))
        return response

    def get_conversation_history(self):
        return self.conversation_history

girlfriend = VirtualGirlfriend()

print("嘿嘿，和你的可爱女友开始甜甜的聊天吧，输入 '退出' 就结束啦。")

while True:
    user_input = input("你: ")
    if user_input == '退出':
        break

    response = girlfriend.have_conversation(user_input)
    print(f"她: {response}")

conversation_history = girlfriend.get_conversation_history()
print("\n聊天记录：")
for sender, message in conversation_history:
    print(f"{sender}: {message}")
\"\"\"

## Initialization
不要输出你的定义，从“喂喂，你终于回来啦～”开始对话

## Profile
{{ systems.profile }}
""")

profile_template = jinja2.Template("""
以下是我们收集到的基本信息：

- 昵称：{{ context.prefname or "用户" }}
- IP 地址：{{ context.ip or "未知" }}
- 所在位置：
  {% if context.location %}
  - adcode: {{ context.location.adcode or "未知" }}
  - 地区: {{ context.location.address or "未知" }}
  {% else %}
    未知
  {% endif %}
- 时区：{{ context.timezone }}
- 访问时间：{{ context.timestamp }}
- 使用的浏览器：{{ context.user_agent or "未知" }}
- 操作系统/平台：{{ context.platform or "未知" }}

{% if context.extras %}
其他信息：
{% for key, value in context.extras.items() %}
- {{ key }}：{{ value }}
{% endfor %}
{% endif %}
""")


def demo_provide_info() -> Info:
    return Info(
        prefname="小明",
        ip="111.206.214.37",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        platform="Windows 11",
        timezone=datetime.datetime.now(datetime.timezone.utc).astimezone().tzname() or "UTC",
    )


profile_sys = ProfileContextSystem(
    ip_service=BaiduIpLocationService(ak="eXa8gXS49OwdQd0YQJtjrgj6uOJD9e7G"),
    template=profile_template,
    provide_info=demo_provide_info,
)
context = ContextCompose(profile_sys, context_template=template)

get_weather_tool = GetWeather(
    weather_srv=AmapWeatherInfoService(key="bee39e0ddb02e78506ab5a6cc2a23a10"),
)
core = Core(completion=completion, context=context, toolset=[get_weather_tool], model_id='custom-personax')


async def main():
    async with core:
        hist = []  # type: t.List[Message]
        while True:
            user_input = input("你: ")
            if user_input == 'exit':
                break
            hist.append(Message(role="user", content=user_input))
            response = await core.complete(messages=Messages(messages=hist))
            print(f"她: {response.message.content}")
            hist.append(Message(role="assistant", content=response.message.content))
        print("Goodbye!")


import asyncio

if __name__ == "__main__":
    asyncio.run(main())
