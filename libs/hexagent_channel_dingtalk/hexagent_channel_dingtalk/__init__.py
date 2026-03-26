"""HexAgent DingTalk Channel.

Connects DingTalk (\u9489\u9489) to HexAgent via Stream mode (no public IP required).

Quick start::

    import asyncio
    from hexagent import create_agent
    from hexagent.computer.local import LocalNativeComputer
    from hexagent.harness.model import ModelProfile
    from langchain_anthropic import ChatAnthropic
    from hexagent_channel_dingtalk import DingTalkChannel, DingTalkConfig

    async def main():
        model = ModelProfile(model=ChatAnthropic(model="claude-opus-4-6"))
        async with await create_agent(model, LocalNativeComputer()) as agent:
            config = DingTalkConfig(
                client_id="your-app-key",
                client_secret="your-app-secret",
            )
            channel = DingTalkChannel(agent=agent, config=config)
            await channel.start()

    asyncio.run(main())
"""

from hexagent_channel_dingtalk.channel import DingTalkChannel
from hexagent_channel_dingtalk.config import DingTalkConfig

__all__ = ["DingTalkChannel", "DingTalkConfig"]
