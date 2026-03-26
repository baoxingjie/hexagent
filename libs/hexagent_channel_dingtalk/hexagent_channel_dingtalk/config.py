"""DingTalk channel configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DingTalkConfig:
    """Configuration for the DingTalk channel.

    Attributes:
        client_id: DingTalk app Client ID (AppKey / clientId).
        client_secret: DingTalk app Client Secret (AppSecret / clientSecret).
        robot_code: Robot code override. Defaults to ``client_id`` when unset.
        require_mention_in_group: When True (default), group messages are only
            processed when the bot is @mentioned.  Has no effect in 1:1 chats.
        max_response_chars: Truncate agent responses longer than this before
            sending.  DingTalk has a ~20 000-character message limit; the
            default of 4000 keeps replies readable.
        debug: Enable verbose debug logging from the dingtalk-stream SDK.
    """

    client_id: str
    client_secret: str
    robot_code: str | None = None
    require_mention_in_group: bool = True
    max_response_chars: int = 4000
    debug: bool = False

    @property
    def effective_robot_code(self) -> str:
        """Robot code used when calling the DingTalk API."""
        return self.robot_code or self.client_id
