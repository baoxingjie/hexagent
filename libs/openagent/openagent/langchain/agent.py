"""LangChain agent factory for OpenAgent.

This module provides the create_agent function that creates an OpenAgent
agent using LangChain's agent infrastructure with runtime modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent as _create_langchain_agent
from langchain.chat_models import init_chat_model

from openagent.config import AgentConfig
from openagent.langchain.middleware import AgentMiddleware
from openagent.prompts import FRESH_SESSION, PromptContext, compose, load
from openagent.runtime import CapabilityRegistry, PermissionGate
from openagent.runtime.skills import SkillResolver
from openagent.tools import SkillTool, WebFetchTool, WebSearchTool, create_cli_tools

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from openagent.computer import Computer
    from openagent.tools.web import FetchProvider, SearchProvider

DEFAULT_MODEL = "openai:gpt-5.2"


async def create_agent(
    computer: Computer,
    *,
    config: AgentConfig | None = None,
    model: str | BaseChatModel | None = None,
    search_provider: SearchProvider | None = None,
    fetch_provider: FetchProvider | None = None,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    system_prompt: str | None = None,
) -> CompiledStateGraph:
    """Create an OpenAgent agent using LangChain.

    OpenAgent agents require a LLM that supports tool calling.

    Default tools: ``bash``, ``read``, ``write``, ``edit``, ``glob``,
    ``grep``.  Web tools (``web_search``, ``web_fetch``) are included
    only when their providers are supplied.  The ``skill`` tool is
    included when skills are discovered from configured search paths.

    Args:
        computer: The Computer instance for CLI tools.
        config: Agent configuration. Defaults to ``AgentConfig()``.
        model: The model to use. Defaults to ``openai:gpt-5``.
        tools: Additional tools the agent should have access to.
        search_provider: Web search provider (e.g. ``TavilySearchProvider()``).
        fetch_provider: Web fetch provider (e.g. ``JinaFetchProvider()``).
        checkpointer: LangGraph checkpointer for conversation persistence.
        store: LangGraph store for cross-thread memory.
        system_prompt: Additional instructions for the agent.

    Returns:
        A configured OpenAgent agent.

    Examples:
        Basic usage with defaults::

            agent = await create_agent(LocalNativeComputer())
            result = await agent.ainvoke({"messages": [...]})

        With skills::

            config = AgentConfig(
                skills=SkillsConfig(search_paths=("/mnt/skills",)),
            )
            agent = await create_agent(computer, config=config)
    """
    config = config or AgentConfig()

    # Initialize model
    if model is None:
        model = init_chat_model(DEFAULT_MODEL)
    elif isinstance(model, str):
        model = init_chat_model(model)

    # Build registry with all default tools
    registry = _create_default_registry(
        computer,
        search_provider=search_provider,
        fetch_provider=fetch_provider,
    )

    # Discover and register skills
    resolver: SkillResolver | None = None
    if config.skills.search_paths:
        resolver = SkillResolver(computer, config.skills.search_paths)
        skills = await resolver.discover()
        for skill in skills:
            registry.register_skill(skill)
        # Register skill tool with discovered skill names
        skill_tool = SkillTool(registered_skills={s.name for s in skills})
        registry.register_tool(skill_tool)

    # Assemble system prompt
    ctx = PromptContext(
        tools=registry.get_tools(),
        skills=registry.get_skills(),
        mcps=registry.get_mcps(),
        user_instructions=system_prompt,
    )
    assembled_prompt = compose(FRESH_SESSION, ctx)

    # Create middleware
    middleware = AgentMiddleware(
        registry=registry,
        permission_gate=PermissionGate(),
        compaction_prompt=load("user_prompt_compaction_request"),
        summary_rebuild_template=load("user_prompt_compaction_summary_rebuild"),
        compaction_threshold=config.compaction.threshold,
        skill_resolver=resolver,
    )

    return _create_langchain_agent(
        model,
        tools=tools,
        system_prompt=assembled_prompt,
        middleware=[middleware],
        checkpointer=checkpointer,
        store=store,
    ).with_config({"recursion_limit": 1000})


def _create_default_registry(
    computer: Computer,
    *,
    search_provider: SearchProvider | None = None,
    fetch_provider: FetchProvider | None = None,
) -> CapabilityRegistry:
    """Create a registry populated with all default tools.

    Args:
        computer: The Computer instance for CLI tools.
        search_provider: Optional web search provider.
        fetch_provider: Optional web fetch provider.

    Returns:
        A CapabilityRegistry with all default tools registered.
    """
    registry = CapabilityRegistry()

    for tool in create_cli_tools(computer):
        registry.register_tool(tool)

    if search_provider is not None:
        registry.register_tool(WebSearchTool(search_provider))

    if fetch_provider is not None:
        registry.register_tool(WebFetchTool(fetch_provider))

    return registry
