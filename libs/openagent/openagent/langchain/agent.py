"""LangChain agent factory for OpenAgent.

This module provides the create_agent function that creates an OpenAgent
agent using LangChain's agent infrastructure with runtime modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent as _create_langchain_agent
from langchain.chat_models import init_chat_model

from openagent.langchain.middleware import AgentMiddleware, ApprovalCallback
from openagent.runtime import (
    CapabilityRegistry,
    ContextManager,
    PermissionGate,
    SystemPromptAssembler,
    SystemReminder,
)
from openagent.tools import create_cli_tools

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.structured_output import ResponseFormat
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.cache.base import BaseCache
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from openagent.computer import Computer

BASE_AGENT_PROMPT = """You are OpenAgent, a general-purpose agent that uses a computer to complete tasks like how human does."""

DEFAULT_MODEL = "openai:gpt-5"


def create_agent(
    computer: Computer,
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    # Runtime module customization
    registry: CapabilityRegistry | None = None,
    context_manager: ContextManager | None = None,
    permission_gate: PermissionGate | None = None,
    system_reminder: SystemReminder | None = None,
    environment: dict[str, str] | None = None,
    approval_callback: ApprovalCallback | None = None,
    # LangChain passthrough
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create an OpenAgent agent using LangChain.

    OpenAgent agents require a LLM that supports tool calling.

    This agent will by default have access to CLI tools:
    `bash`, `read`, `write`, `edit`, `glob`, `grep`.

    All CLI tools share a persistent bash session where state (working directory,
    environment variables, shell functions) persists across tool calls.

    Args:
        computer: The Computer instance for CLI tools.
            Can be `LocalNativeComputer` for local execution or `RemoteE2BComputer`
            for cloud sandbox execution.
        model: The model to use. Defaults to `openai:gpt-5`.
            Use the `provider:model` format (e.g., `openai:gpt-5.2`) to quickly switch.
        tools: Additional tools the agent should have access to.
            In addition to custom tools, OpenAgent agents include built-in CLI tools.
        system_prompt: Additional instructions for the agent. Will be included in
            the system prompt.
        registry: Optional CapabilityRegistry for managing tools/skills.
            If not provided, a default registry with CLI tools is created.
        context_manager: Optional ContextManager for compaction threshold checks.
            If not provided, a default manager (100K token threshold) is created.
        permission_gate: Optional PermissionGate for safety validation.
            If not provided, a default gate (allow all) is created.
        system_reminder: Optional SystemReminder for injecting contextual reminders
            into user messages based on conditions.
        environment: Optional environment context (key-value pairs) to include in
            the system prompt. Useful for providing platform, cwd, etc.
        approval_callback: Optional callback for human-in-the-loop approval of
            tool calls that return NEEDS_APPROVAL from the permission gate.
        response_format: A structured output response format for the agent.
        context_schema: The schema of the agent context.
        checkpointer: Optional Checkpointer for persisting agent state.
        store: Optional store for LangGraph runtime.
        debug: Whether to enable debug mode.
        name: The name of the agent.
        cache: The cache to use for the agent.

    Returns:
        A configured OpenAgent agent.

    Examples:
        Basic usage with defaults:
        ```python
        from openagent import create_agent
        from openagent.computer import LocalNativeComputer

        agent = create_agent(LocalNativeComputer())
        result = await agent.ainvoke({"messages": [...]})
        ```

        With custom runtime modules:
        ```python
        from openagent import create_agent, CapabilityRegistry, PermissionGate
        from openagent.computer import LocalNativeComputer

        registry = CapabilityRegistry()
        # Register custom tools...

        gate = PermissionGate()
        # Register safety rules...

        agent = create_agent(
            LocalNativeComputer(),
            registry=registry,
            permission_gate=gate,
        )
        ```
    """
    # Initialize model
    if model is None:
        model = init_chat_model(DEFAULT_MODEL)
    elif isinstance(model, str):
        model = init_chat_model(model)

    # Create default runtime modules if not provided
    if registry is None:
        registry = _create_default_registry(computer)

    if context_manager is None:
        context_manager = ContextManager()

    if permission_gate is None:
        permission_gate = PermissionGate()

    # Create prompt assembler (always new, stateless)
    assembler = SystemPromptAssembler()

    # Create the middleware that wires everything together
    middleware = AgentMiddleware(
        registry=registry,
        assembler=assembler,
        context_manager=context_manager,
        permission_gate=permission_gate,
        base_prompt=BASE_AGENT_PROMPT,
        user_instructions=system_prompt,
        system_reminder=system_reminder,
        environment=environment,
        approval_callback=approval_callback,
    )

    # Create the LangChain agent
    return _create_langchain_agent(
        model,
        tools=tools,
        middleware=[middleware],
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})


def _create_default_registry(computer: Computer) -> CapabilityRegistry:
    """Create a default registry with CLI tools.

    Args:
        computer: The Computer instance for CLI tools.

    Returns:
        A CapabilityRegistry with CLI tools registered.
    """
    registry = CapabilityRegistry()

    # Register all CLI tools
    for tool in create_cli_tools(computer):
        registry.register_tool(tool)

    return registry
