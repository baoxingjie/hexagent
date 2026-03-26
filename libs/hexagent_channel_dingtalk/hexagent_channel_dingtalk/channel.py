"""DingTalk channel for HexAgent.

Architecture
------------
The ``dingtalk-stream`` Python SDK runs its own event loop inside
``start_forever()``.  We launch it in a daemon thread so that the caller's
asyncio event loop (which runs the agent) stays free.

When a DingTalk message arrives the SDK calls ``_RobotHandler.process()``
inside *its* loop.  We use ``asyncio.run_coroutine_threadsafe`` to schedule
``DingTalkChannel._handle_message()`` on the *caller's* loop, ack the message
immediately, and let the agent work asynchronously.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import TYPE_CHECKING, Any

from hexagent_channel_dingtalk.config import DingTalkConfig
from hexagent_channel_dingtalk.send_service import (
    send_markdown_to_group,
    send_markdown_to_private,
    send_text_to_group,
    send_text_to_private,
)

if TYPE_CHECKING:
    from hexagent import Agent

logger = logging.getLogger(__name__)


class DingTalkChannel:
    """Wire a HexAgent agent to a DingTalk robot via Stream mode.

    The channel connects to DingTalk using the Stream WebSocket protocol
    (no public IP required).  Incoming messages are forwarded to the
    wrapped ``Agent``; the agent's reply is sent back as either plain text
    or Markdown depending on whether the response contains Markdown syntax.

    Conversation history is scoped by DingTalk ``conversationId``, which is
    passed as LangGraph's ``thread_id``.  If the agent was created with a
    persistent checkpointer, each conversation will remember prior context
    across separate messages.

    Args:
        agent: A HexAgent ``Agent`` instance (from ``create_agent()``).
        config: DingTalk credentials and behaviour options.

    Example::

        channel = DingTalkChannel(
            agent=agent,
            config=DingTalkConfig(
                client_id=os.environ["DINGTALK_CLIENT_ID"],
                client_secret=os.environ["DINGTALK_CLIENT_SECRET"],
            ),
        )
        await channel.start()  # blocks until channel.stop() is called
    """

    def __init__(self, agent: Agent, config: DingTalkConfig) -> None:
        self._agent = agent
        self._config = config
        self._sdk_client: Any | None = None
        self._stop_event: asyncio.Event | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the DingTalk Stream client and block until :meth:`stop` is called.

        Raises:
            RuntimeError: If ``dingtalk-stream`` is not installed.
        """
        try:
            import dingtalk_stream  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "dingtalk-stream is required: pip install dingtalk-stream"
            ) from exc

        loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()

        channel = self  # capture for nested class

        class _RobotHandler(dingtalk_stream.ChatbotHandler):
            """Callback handler that dispatches to the main asyncio loop."""

            async def process(  # type: ignore[override]
                self, callback: Any
            ) -> tuple[str, str]:
                try:
                    incoming = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
                    # Schedule handling in the caller's event loop; do NOT
                    # await here — we ack immediately and process async.
                    asyncio.run_coroutine_threadsafe(
                        channel._handle_message(incoming), loop
                    )
                except Exception:
                    logger.exception("Error dispatching DingTalk message to main loop")
                return dingtalk_stream.AckMessage.STATUS_OK, "OK"

        credential = dingtalk_stream.Credential(
            self._config.client_id, self._config.client_secret
        )
        sdk_client = dingtalk_stream.DingTalkStreamClient(
            credential, logger_level=logging.DEBUG if self._config.debug else logging.WARNING
        )
        sdk_client.register_callback_handler(
            dingtalk_stream.ChatbotHandler.TOPIC, _RobotHandler()
        )
        self._sdk_client = sdk_client

        logger.info(
            "Starting DingTalk Stream client (clientId=%s)", self._config.client_id
        )

        sdk_thread = threading.Thread(
            target=sdk_client.start_forever, daemon=True, name="dingtalk-stream"
        )
        sdk_thread.start()

        # Block the caller until stop() is called.
        await self._stop_event.wait()
        logger.info("DingTalk channel stopped")

    def stop(self) -> None:
        """Signal the channel to stop.

        Safe to call from any thread or coroutine.  The :meth:`start` coroutine
        will return shortly after this is called.
        """
        if self._stop_event is not None and not self._stop_event.is_set():
            # _stop_event lives in an asyncio event loop; set it thread-safely.
            try:
                loop = self._stop_event._loop  # type: ignore[attr-defined]
                loop.call_soon_threadsafe(self._stop_event.set)
            except Exception:
                logger.debug("stop() called before event loop was attached", exc_info=True)

    # ------------------------------------------------------------------ #
    # Message handling
    # ------------------------------------------------------------------ #

    async def _handle_message(self, incoming: Any) -> None:
        """Process one inbound DingTalk message end-to-end."""
        text: str = (incoming.text.content or "").strip() if incoming.text else ""
        is_group: bool = str(incoming.conversation_type) == "2"

        if is_group:
            if self._config.require_mention_in_group:
                # Only respond when the bot is @mentioned
                at_users: list = incoming.at_users or []
                bot_id: str = incoming.chatbot_user_id or ""
                mentioned = any(
                    (u.get("dingtalkId") or u.get("staffId") or "") == bot_id
                    for u in at_users
                )
                if not mentioned:
                    logger.debug("Group message ignored (bot not @mentioned)")
                    return
            # Strip leading @mentions from the text before passing to agent
            text = re.sub(r"^(@\S+\s*)+", "", text).strip()

        if not text:
            logger.debug("Ignoring message with empty text")
            return

        conversation_id: str = incoming.conversation_id or ""
        sender_staff_id: str = incoming.sender_staff_id or ""
        sender_id: str = incoming.sender_id or ""
        open_conversation_id: str = getattr(incoming, "open_conversation_id", "") or ""

        logger.info(
            "Message received: conv=%s sender=%s group=%s text=%r",
            conversation_id,
            sender_staff_id or sender_id,
            is_group,
            text[:80],
        )

        response_text = await self._invoke_agent(text, thread_id=conversation_id)
        if not response_text:
            logger.warning("Agent returned empty response for conv=%s", conversation_id)
            return

        # Truncate to DingTalk's practical message limit
        if len(response_text) > self._config.max_response_chars:
            response_text = (
                response_text[: self._config.max_response_chars - 3] + "..."
            )

        use_markdown = _looks_like_markdown(response_text)

        try:
            if is_group:
                conv_id = open_conversation_id or conversation_id
                if use_markdown:
                    await send_markdown_to_group(
                        self._config, conv_id, title="Reply", text=response_text
                    )
                else:
                    await send_text_to_group(self._config, conv_id, response_text)
            else:
                target = sender_staff_id or sender_id
                if not target:
                    logger.error(
                        "Cannot send reply: no staffId/senderId in message conv=%s",
                        conversation_id,
                    )
                    return
                if use_markdown:
                    await send_markdown_to_private(
                        self._config, target, title="Reply", text=response_text
                    )
                else:
                    await send_text_to_private(self._config, target, response_text)

            logger.info(
                "Reply sent: conv=%s len=%d", conversation_id, len(response_text)
            )
        except Exception:
            logger.exception(
                "Failed to send reply to DingTalk: conv=%s", conversation_id
            )

    async def _invoke_agent(self, text: str, *, thread_id: str) -> str | None:
        """Run the HexAgent agent and return the final reply text.

        Uses ``ainvoke`` (not streaming) so the full response is available
        before we send the DingTalk message.  The ``thread_id`` is forwarded
        to LangGraph's checkpointer so conversation context is preserved.

        Args:
            text: The user's message text.
            thread_id: Used as the LangGraph ``thread_id`` for state persistence.

        Returns:
            The agent's reply as a plain string, or ``None`` if no AI message
            was found in the result.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        input_dict: dict[str, Any] = {"messages": [HumanMessage(content=text)]}
        run_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10_000,
        }

        try:
            result = await self._agent.ainvoke(input_dict, config=run_config)
        except Exception:
            logger.exception("Agent invocation failed for thread_id=%s", thread_id)
            return None

        messages: list[Any] = result.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        if last_ai is None:
            return None

        content = last_ai.content
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            # Structured content blocks (e.g. Anthropic tool-use format)
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts).strip() or None
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_MD_PATTERN = re.compile(
    r"(^#{1,6}\s|\*\*|__|\[.+?\]\(.+?\)|^[-*+]\s|^\d+\.\s|```|`[^`]+`)",
    re.MULTILINE,
)


def _looks_like_markdown(text: str) -> bool:
    """Heuristic: return True if ``text`` contains common Markdown syntax."""
    return bool(_MD_PATTERN.search(text))
