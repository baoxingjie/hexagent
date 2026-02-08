"""Agent middleware for wiring runtime modules to LangChain.

This middleware coordinates the runtime modules (CapabilityRegistry,
CompactionController, PermissionGate) with LangChain's agent infrastructure.

All compaction decisions are delegated to the framework-agnostic
CompactionController. This middleware translates the returned ContextUpdate
into LangChain-specific message operations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware as LangChainAgentMiddleware,
)
from langchain.agents.middleware.types import (
    AgentState,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Overwrite as LangGraphOverwrite

from openagent.langchain.adapter import to_langchain_tool
from openagent.runtime import (
    CapabilityRegistry,
    CompactionController,
    CompactionPhase,
    PermissionGate,
    PermissionResult,
)
from openagent.runtime.context import (
    Append,
    ContextUpdate,
    Overwrite,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from openagent.runtime.skills import SkillResolver

# Type alias for token counter function
TokenCounter = Callable[[Sequence[BaseMessage]], int]


class ApprovalCallback(Protocol):
    """Callback for human-in-the-loop approval of tool calls.

    Implement this protocol to provide custom approval logic for
    tools that return NEEDS_APPROVAL from the permission gate.

    Examples:
        ```python
        async def cli_approval(
            tool_name: str,
            tool_args: dict[str, Any],
            approval_prompt: str | None,
        ) -> bool:
            response = input(f"Allow {tool_name}? [y/n]: ")
            return response.lower() == "y"


        middleware = AgentMiddleware(
            ...,
            approval_callback=cli_approval,
        )
        ```
    """

    async def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        approval_prompt: str | None,
    ) -> bool:
        """Request approval for a tool call.

        Args:
            tool_name: Name of the tool requesting approval.
            tool_args: Arguments passed to the tool.
            approval_prompt: Optional prompt describing why approval is needed.

        Returns:
            True if approved, False if denied.
        """
        ...


# Average characters per token (rough estimate for English text)
_CHARS_PER_TOKEN = 4

# Minimum messages required before compaction can trigger.
# A single summary message after compaction should not re-trigger.
_MIN_MESSAGES_FOR_COMPACTION = 2


def _estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """Estimate token count from messages.

    Uses a simple heuristic: ~4 characters per token for English text.

    Args:
        messages: The messages to estimate tokens for.

    Returns:
        Estimated token count.
    """
    total_chars = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    total_chars += len(block)
                elif isinstance(block, dict) and "text" in block:
                    total_chars += len(str(block["text"]))
    return total_chars // _CHARS_PER_TOKEN


class AgentMiddleware(LangChainAgentMiddleware):
    """Middleware that wires runtime modules to LangChain agent hooks.

    Delegates compaction decisions to the framework-agnostic
    ``CompactionController`` and translates the returned ``ContextUpdate``
    into LangChain message operations.

    Coordinates:
    - CapabilityRegistry: Provides tools to the agent
    - CompactionController: 3-phase stateless compaction protocol
    - PermissionGate: Validates tool calls with optional human-in-the-loop

    Hook mapping:
    - tools property: Provides tools from CapabilityRegistry
    - abefore_model: Applies CompactionController.pre_model_update()
    - aafter_model: Applies CompactionController.post_model_transition()
    - awrap_tool_call: Validates via PermissionGate, calls approval callback

    Examples:
        Basic usage::

            middleware = AgentMiddleware(
                registry=registry,
                permission_gate=PermissionGate(),
                compaction_prompt=library.get("compaction/request"),
                summary_rebuild_template=library.get("compaction/summary_rebuild"),
            )
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        permission_gate: PermissionGate,
        *,
        compaction_prompt: str,
        summary_rebuild_template: str,
        compaction_threshold: int = 100_000,
        count_tokens: TokenCounter | None = None,
        approval_callback: ApprovalCallback | None = None,
        skill_resolver: SkillResolver | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            registry: The capability registry providing tools.
            permission_gate: The permission gate for validating tool calls.
            compaction_prompt: Prompt for requesting conversation summaries.
                Canonical source: ``user_prompt_compaction_request`` via
                ``load()``.
            summary_rebuild_template: Template for rebuilding context after
                compaction.  Must contain ``${SUMMARY_CONTENT}`` placeholder.
                Canonical source: ``user_prompt_compaction_summary_rebuild``
                via ``load()``.
            compaction_threshold: Token count that triggers context compaction.
            count_tokens: Optional function to count tokens in messages. If not
                provided, uses character-based estimation (~4 chars/token).
            approval_callback: Optional callback for human-in-the-loop approval.
            skill_resolver: Optional skill resolver for injecting skill content.
        """
        self._registry = registry
        self._permission_gate = permission_gate
        self._approval_callback = approval_callback
        self._count_tokens = count_tokens or _estimate_tokens
        self._summary_rebuild_template = summary_rebuild_template
        self._skill_resolver = skill_resolver

        self._compaction = CompactionController(
            compaction_prompt,
            threshold=compaction_threshold,
        )

        # Cache for converted LangChain tools
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> Sequence[BaseTool]:  # type: ignore[override]
        """Get tools from the registry as LangChain tools.

        Returns:
            Sequence of LangChain BaseTool instances.
        """
        if self._tools_cache is None:
            self._tools_cache = [to_langchain_tool(tool) for tool in self._registry.get_tools()]
        return self._tools_cache

    # --- Async hooks (primary implementations) ---

    async def abefore_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Apply skill injection or compaction updates before model call.

        Skill injection takes priority: if the most recent tool call was
        a ``skill`` invocation, the skill's markdown content is appended
        as a ``HumanMessage``.

        Otherwise delegates to ``CompactionController.pre_model_update()``
        and translates the returned ``ContextUpdate`` into LangChain
        message operations.

        Args:
            state: The current agent state containing messages.
            _runtime: The LangGraph runtime context.

        Returns:
            State updates dict, or None if no changes.
        """
        # --- Skill injection (takes priority over compaction) ---
        if self._skill_resolver is not None:
            skill_name = self._detect_skill_call(state["messages"])
            if skill_name is not None:
                try:
                    content = await self._skill_resolver.load_content(skill_name)
                except (KeyError, RuntimeError):
                    content = None
                if content is not None:
                    phase = CompactionPhase(state.get("compaction_phase", CompactionPhase.NONE))
                    skill_update = Append(content=content)
                    return self._apply_context_update(state["messages"], skill_update, phase)

        # --- Compaction (existing logic, unchanged) ---
        phase = CompactionPhase(state.get("compaction_phase", CompactionPhase.NONE))
        update, new_phase = self._compaction.pre_model_update(phase)

        if update is not None:
            return self._apply_context_update(state["messages"], update, new_phase)

        return None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Check for compaction after model response.

        Delegates to ``CompactionController.post_model_transition()`` and
        returns phase transition with jump-to-model if needed.

        Args:
            state: The current agent state with messages.
            _runtime: The LangGraph runtime context.

        Returns:
            State updates dict with phase transition and jump, or None.
        """
        messages = state["messages"]
        phase = CompactionPhase(state.get("compaction_phase", CompactionPhase.NONE))

        # Need at least 2 messages to compact; a single summary message
        # after compaction should not re-trigger the cycle.
        if phase == CompactionPhase.NONE and len(messages) < _MIN_MESSAGES_FOR_COMPACTION:
            return None

        token_count = self._count_tokens(messages)
        should_rerun, new_phase = self._compaction.post_model_transition(token_count, phase)

        if should_rerun:
            return {"compaction_phase": new_phase, "jump_to": "model"}

        return None

    async def awrap_tool_call(  # type: ignore[override]
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Check permission before tool execution.

        Args:
            request: The tool call request.
            handler: The async handler to execute the tool.

        Returns:
            The tool message response, or an error response if denied.
        """
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})

        decision = await self._permission_gate.check(tool_name, tool_args)

        if decision.result == PermissionResult.DENIED:
            return self._create_denied_response(request, decision.reason)

        if decision.result == PermissionResult.NEEDS_APPROVAL:
            if self._approval_callback is None:
                return self._create_denied_response(
                    request,
                    f"Action requires approval: {decision.approval_prompt or 'No details provided'}",
                )

            approved = await self._approval_callback(
                tool_name,
                tool_args,
                decision.approval_prompt,
            )

            if not approved:
                return self._create_denied_response(request, "Action denied by user")

        return await handler(request)

    # --- Private helpers ---

    def _detect_skill_call(
        self,
        messages: Sequence[BaseMessage],
    ) -> str | None:
        """Detect if the most recent tool call was 'skill' and return the skill name.

        Walks messages backwards to find the last ToolMessage, then checks
        the corresponding AIMessage for a 'skill' tool call.

        Returns None if:
        - No recent skill tool call found
        - A HumanMessage already follows the skill ToolMessage (already injected)
        """
        if not messages:
            return None

        # Walk backwards: find the last ToolMessage
        # If we hit a HumanMessage first, skill was already injected
        last_tool_msg_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, HumanMessage):
                return None
            if isinstance(msg, ToolMessage):
                last_tool_msg_idx = i
                break

        if last_tool_msg_idx is None:
            return None

        tool_msg = messages[last_tool_msg_idx]
        if not isinstance(tool_msg, ToolMessage):
            return None  # unreachable, satisfies mypy
        tool_call_id = tool_msg.tool_call_id

        # Find the corresponding AIMessage with tool_calls
        for i in range(last_tool_msg_idx - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("id") == tool_call_id and tc.get("name") == "skill":
                        args: dict[str, str] = tc.get("args", {})
                        return args.get("skill")
                break  # Only check the nearest AIMessage

        return None

    def _apply_context_update(
        self,
        messages: Sequence[BaseMessage],
        update: ContextUpdate,
        new_phase: CompactionPhase,
    ) -> dict[str, Any]:
        """Translate a ContextUpdate into LangChain state updates.

        Args:
            messages: The current message history.
            update: The context update to apply.
            new_phase: The new compaction phase after this update.

        Returns:
            State updates dict with modified messages and phase.
        """
        if isinstance(update, Overwrite):
            rebuilt = self._rebuild_with_summary(messages)
            return {
                "messages": LangGraphOverwrite(rebuilt),
                "compaction_phase": new_phase,
            }

        if isinstance(update, Append):
            msgs = list(messages)
            msg_cls = HumanMessage if update.role == "user" else AIMessage
            msgs.append(msg_cls(content=update.content))
            return {"messages": msgs}

        # Inject
        msgs = list(messages)
        last = msgs[-1]
        if isinstance(last.content, str):
            content = last.content
        elif isinstance(last.content, list):
            content = "".join(block if isinstance(block, str) else block.get("text", "") for block in last.content if isinstance(block, (str, dict)))
        else:
            content = str(last.content)
        new_content = f"{update.content}\n\n{content}" if update.position == "prepend" else f"{content}\n\n{update.content}"
        kwargs: dict[str, Any] = {"content": new_content}
        if last.id is not None:
            kwargs["id"] = last.id
        if isinstance(last, ToolMessage):
            kwargs["tool_call_id"] = last.tool_call_id
        msgs[-1] = last.__class__(**kwargs)
        return {"messages": msgs}

    def _rebuild_with_summary(
        self,
        messages: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        """Extract summary from last AIMessage and rebuild message history.

        Args:
            messages: Current message history including summary response.

        Returns:
            Rebuilt message list with summary as initial context.
        """
        summary_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str):
                    summary_content = content
                elif isinstance(content, list):
                    summary_content = "".join(
                        block if isinstance(block, str) else block.get("text", "") for block in content if isinstance(block, (str, dict))
                    )
                break

        rebuilt = self._summary_rebuild_template.replace("${SUMMARY_CONTENT}", summary_content)
        return [
            HumanMessage(content=rebuilt),
        ]

    def _create_denied_response(
        self,
        request: ToolCallRequest,
        reason: str | None,
    ) -> ToolMessage:
        """Create a tool response for a denied action."""
        error_message = f"Permission denied: {reason}" if reason else "Permission denied"
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call.get("id", ""),
        )

    # --- Sync hooks (for interface compliance) ---

    def before_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Sync version of abefore_model.

        Note: Skill injection requires async (Computer I/O) and is only
        available via the async ``abefore_model`` hook.
        """
        phase = CompactionPhase(state.get("compaction_phase", CompactionPhase.NONE))
        update, new_phase = self._compaction.pre_model_update(phase)

        if update is not None:
            return self._apply_context_update(state["messages"], update, new_phase)

        return None

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Sync version of aafter_model."""
        messages = state["messages"]
        phase = CompactionPhase(state.get("compaction_phase", CompactionPhase.NONE))

        if phase == CompactionPhase.NONE and len(messages) < _MIN_MESSAGES_FOR_COMPACTION:
            return None

        token_count = self._count_tokens(messages)
        should_rerun, new_phase = self._compaction.post_model_transition(token_count, phase)

        if should_rerun:
            return {"compaction_phase": new_phase, "jump_to": "model"}

        return None

    def wrap_tool_call(  # type: ignore[override]
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Sync stub. Use awrap_tool_call for full functionality."""
        return handler(request)
