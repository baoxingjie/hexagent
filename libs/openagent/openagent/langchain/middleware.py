"""Agent middleware for wiring runtime modules to LangChain.

This middleware coordinates the runtime modules (CapabilityRegistry,
SystemPromptAssembler, ContextManager, PermissionGate, SystemReminder)
with LangChain's agent infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware as LangChainAgentMiddleware,
)
from langchain.agents.middleware.types import (
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from openagent.langchain.adapter import to_langchain_tool
from openagent.runtime import (
    CapabilityRegistry,
    ContextManager,
    PermissionGate,
    PermissionResult,
    SystemPromptAssembler,
    SystemReminder,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime


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


def _estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """Estimate token count from messages.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is a rough estimate, not exact tokenization.

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
            # Handle content blocks (list of dicts or strings)
            for block in content:
                if isinstance(block, str):
                    total_chars += len(block)
                elif isinstance(block, dict) and "text" in block:
                    total_chars += len(str(block["text"]))
    return total_chars // _CHARS_PER_TOKEN


class AgentMiddleware(LangChainAgentMiddleware):
    """Middleware that wires runtime modules to LangChain agent hooks.

    This middleware is designed for async-first usage (ainvoke, astream).
    Sync hooks are provided only as pass-through stubs for interface compliance.

    Coordinates:
    - CapabilityRegistry: Provides tools to the agent
    - SystemPromptAssembler: Builds system prompts from typed data
    - ContextManager: Injects compaction prompt when token threshold exceeded
    - PermissionGate: Validates tool calls with optional human-in-the-loop
    - SystemReminder: Injects contextual reminders into user messages

    Hook mapping:
    - tools property: Provides tools from CapabilityRegistry
    - abefore_model: Injects reminders and compaction prompt
    - awrap_model_call: Injects system prompt
    - awrap_tool_call: Validates via PermissionGate, calls approval callback

    Examples:
        Basic usage:
        ```python
        from openagent.langchain import AgentMiddleware
        from openagent.runtime import (
            CapabilityRegistry,
            SystemPromptAssembler,
            ContextManager,
            PermissionGate,
        )

        registry = CapabilityRegistry()
        registry.register_tool(bash_tool)

        middleware = AgentMiddleware(
            registry=registry,
            assembler=SystemPromptAssembler(),
            context_manager=ContextManager(),
            permission_gate=PermissionGate(),
            base_prompt="You are a helpful assistant.",
        )
        ```

        With reminders and approval callback:
        ```python
        from openagent.runtime import SystemReminder

        reminder = SystemReminder()
        reminder.add(
            condition=lambda s: len(s["messages"]) > 10,
            text="Consider summarizing progress so far.",
        )


        async def approve(name, args, prompt):
            return input(f"Allow {name}? [y/n]: ").lower() == "y"


        middleware = AgentMiddleware(
            registry=registry,
            assembler=SystemPromptAssembler(),
            context_manager=ContextManager(),
            permission_gate=permission_gate,
            base_prompt="You are a helpful assistant.",
            system_reminder=reminder,
            environment={"platform": "darwin", "cwd": "/project"},
            approval_callback=approve,
        )
        ```
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        assembler: SystemPromptAssembler,
        context_manager: ContextManager,
        permission_gate: PermissionGate,
        base_prompt: str,
        user_instructions: str | None = None,
        system_reminder: SystemReminder | None = None,
        environment: dict[str, str] | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            registry: The capability registry providing tools.
            assembler: The system prompt assembler.
            context_manager: The context manager for compaction decisions.
            permission_gate: The permission gate for validating tool calls.
            base_prompt: The base agent prompt/persona.
            user_instructions: Optional additional user instructions.
            system_reminder: Optional SystemReminder for injecting contextual
                reminders into messages based on conditions.
            environment: Optional environment context (key-value pairs) to include
                in the system prompt.
            approval_callback: Optional callback for human-in-the-loop approval
                of tool calls that return NEEDS_APPROVAL.
        """
        self._registry = registry
        self._assembler = assembler
        self._context_manager = context_manager
        self._permission_gate = permission_gate
        self._base_prompt = base_prompt
        self._user_instructions = user_instructions
        self._system_reminder = system_reminder
        self._environment = environment
        self._approval_callback = approval_callback

        # Cache for converted LangChain tools
        self._tools_cache: list[BaseTool] | None = None

        # Track if compaction prompt was injected this session
        self._compaction_injected = False

    @property
    def tools(self) -> Sequence[BaseTool]:  # type: ignore[override]
        """Get tools from the registry as LangChain tools.

        Tools are converted lazily and cached.

        Returns:
            Sequence of LangChain BaseTool instances.
        """
        if self._tools_cache is None:
            self._tools_cache = [to_langchain_tool(tool) for tool in self._registry.get_tools()]
        return self._tools_cache

    def _build_system_prompt(self, existing_prompt: str | None) -> str:
        """Build the complete system prompt.

        Args:
            existing_prompt: Any existing system prompt from the request.

        Returns:
            The assembled system prompt.
        """
        assembled = self._assembler.assemble(
            base=self._base_prompt,
            tools=self._registry.get_tools(),
            skills=self._registry.get_skills(),
            mcps=self._registry.get_mcps(),
            environment=self._environment,
            user_instructions=self._user_instructions,
        )

        # Prepend any existing prompt from the request
        if existing_prompt:
            return f"{existing_prompt}\n\n{assembled}"
        return assembled

    def _check_compaction(self, messages: Sequence[BaseMessage]) -> bool:
        """Check if context compaction is needed.

        Args:
            messages: The current message history.

        Returns:
            True if compaction threshold is exceeded.
        """
        token_estimate = _estimate_tokens(messages)
        return self._context_manager.needs_compaction(token_estimate)

    # --- Async hooks (primary implementations) ---

    async def abefore_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Inject reminders and compaction prompt before model call.

        This hook runs before each LLM call and can modify the message
        history. It handles:

        1. Compaction prompt injection: When token count exceeds threshold,
           injects a message asking the agent to summarize (once per session).

        2. Reminder injection: If SystemReminder is configured, checks
           conditions and augments the last user message with triggered
           reminders.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime context.

        Returns:
            State updates dict with modified messages, or None if no changes.
        """
        messages = list(state["messages"])
        modified = False

        # 1. Check compaction and inject prompt if needed (once per session)
        if not self._compaction_injected and self._check_compaction(messages):
            self._compaction_injected = True
            compaction_msg = HumanMessage(
                content=self._context_manager.compaction_prompt,
            )
            messages.append(compaction_msg)
            modified = True

        # 2. Inject reminders if configured
        if self._system_reminder is not None:
            # Find the last HumanMessage to augment
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, HumanMessage):
                    content = msg.content
                    if isinstance(content, str):
                        augmented = self._system_reminder.inject(content, state)
                        if augmented != content:
                            messages[i] = HumanMessage(
                                content=augmented,
                                id=getattr(msg, "id", None),
                            )
                            modified = True
                    break

        return {"messages": messages} if modified else None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject system prompt before model call.

        Args:
            request: The model request being processed.
            handler: The async handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        system_prompt = self._build_system_prompt(request.system_prompt)
        updated_request = request.override(system_prompt=system_prompt)  # type: ignore[call-arg]
        return await handler(updated_request)

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

        # Check permission
        decision = await self._permission_gate.check(tool_name, tool_args)

        if decision.result == PermissionResult.DENIED:
            return self._create_denied_response(request, decision.reason)

        if decision.result == PermissionResult.NEEDS_APPROVAL:
            if self._approval_callback is None:
                # No callback provided, treat as denied
                return self._create_denied_response(
                    request,
                    f"Action requires approval: {decision.approval_prompt or 'No details provided'}",
                )

            # Request approval via callback
            approved = await self._approval_callback(
                tool_name,
                tool_args,
                decision.approval_prompt,
            )

            if not approved:
                return self._create_denied_response(request, "Action denied by user")

        # Permission granted (or approved), proceed with execution
        return await handler(request)

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

    # --- Sync hooks (stubs for interface compliance) ---

    def before_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Sync version of abefore_model.

        Implements the same logic as abefore_model for sync invocation.
        Note: This is provided for sync API compatibility, but async API
        is preferred for full functionality.
        """
        messages = list(state["messages"])
        modified = False

        # Check compaction and inject prompt if needed (once per session)
        if not self._compaction_injected and self._check_compaction(messages):
            self._compaction_injected = True
            compaction_msg = HumanMessage(
                content=self._context_manager.compaction_prompt,
            )
            messages.append(compaction_msg)
            modified = True

        # Inject reminders if configured
        if self._system_reminder is not None:
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, HumanMessage):
                    content = msg.content
                    if isinstance(content, str):
                        augmented = self._system_reminder.inject(content, state)
                        if augmented != content:
                            messages[i] = HumanMessage(
                                content=augmented,
                                id=getattr(msg, "id", None),
                            )
                            modified = True
                    break

        return {"messages": messages} if modified else None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync stub. Use awrap_model_call for full functionality."""
        return handler(request)

    def wrap_tool_call(  # type: ignore[override]
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Sync stub. Use awrap_tool_call for full functionality."""
        return handler(request)
