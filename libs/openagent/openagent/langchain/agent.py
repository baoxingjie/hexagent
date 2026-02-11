"""LangChain agent factory for OpenAgent.

Creates an OpenAgent agent using LangChain's agent infrastructure.
No CapabilityRegistry — tools are plain lists. System prompt is managed
in state["messages"] by the middleware, not by LangChain's auto-prepend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent as _create_langchain_agent
from langchain.chat_models import init_chat_model

from openagent.config import AgentConfig
from openagent.harness import PermissionGate, Reminder, SkillResolver, initial_available_skills
from openagent.langchain.middleware import AgentMiddleware
from openagent.prompts import FRESH_SESSION, RESUMED_SESSION, compose, load, substitute
from openagent.tools import SkillTool, WebFetchTool, WebSearchTool, create_cli_tools
from openagent.types import AgentContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from openagent.computer import Computer
    from openagent.tools.base import BaseAgentTool
    from openagent.tools.web import FetchProvider, SearchProvider
    from openagent.types import Skill

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
    reminders: Sequence[Reminder] = (),
) -> CompiledStateGraph[Any]:
    """Create an OpenAgent agent using LangChain.

    OpenAgent agents require a LLM that supports tool calling.

    Default tools: ``bash``, ``read``, ``write``, ``edit``, ``glob``,
    ``grep``.  Web tools (``web_search``, ``web_fetch``) are included
    only when their providers are supplied.  The ``skill`` tool is
    included when skills are discovered from configured search paths.

    Args:
        computer: The Computer instance for CLI tools.
        config: Agent configuration. Defaults to ``AgentConfig()``.
        model: The model to use. Defaults to ``openai:gpt-5.2``.
        search_provider: Web search provider.
        fetch_provider: Web fetch provider.
        tools: Additional tools the agent should have access to.
        checkpointer: LangGraph checkpointer for conversation persistence.
        store: LangGraph store for cross-thread memory.
        system_prompt: Additional instructions for the agent.
        reminders: Reminder rules for dynamic system-reminder injection.

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
    resolved_model = _resolve_model(model)

    # Build default tools
    default_tools: list[BaseAgentTool[Any]] = list(create_cli_tools(computer))
    if search_provider is not None:
        default_tools.append(WebSearchTool(search_provider))
    if fetch_provider is not None:
        default_tools.append(WebFetchTool(fetch_provider))

    # Discover skills
    resolver: SkillResolver | None = None
    skills: list[Skill] = []
    if config.skills.search_paths:
        resolver = SkillResolver(computer, config.skills.search_paths)
        skills = await resolver.discover()
        default_tools.append(SkillTool(catalog=resolver))

    # Build built-in reminders
    builtin_reminders: list[Reminder] = []
    if skills:
        builtin_reminders.append(Reminder(rule=initial_available_skills, position="prepend"))
    all_reminders = [*builtin_reminders, *reminders]

    # Assemble initial system prompt
    ctx = AgentContext(
        tools=default_tools,
        skills=skills,
        mcps=[],
        user_instructions=system_prompt,
    )
    assembled_prompt = compose(FRESH_SESSION, ctx)

    # Create rebuild callback for compaction
    rebuild = _make_rebuild_callback(
        resolver=resolver,
        tools=default_tools,
        summary_template=load("user_prompt_compaction_summary_rebuild"),
        user_instructions=system_prompt,
    )

    # Create middleware (THE runtime)
    middleware = AgentMiddleware(
        tools=default_tools,
        system_prompt=assembled_prompt,
        skills=skills,
        permission_gate=PermissionGate(),
        compaction_prompt=load("user_prompt_compaction_request"),
        compaction_threshold=config.compaction.threshold,
        skill_resolver=resolver,
        reminders=all_reminders,
        rebuild_callback=rebuild,
    )

    # NOTE: system_prompt=None — we manage the SystemMessage in state["messages"]
    # via the middleware's first-turn injection and compaction rebuild.
    return _create_langchain_agent(
        resolved_model,
        tools=tools,
        middleware=[middleware],
        checkpointer=checkpointer,
        store=store,
    ).with_config({"recursion_limit": 1000})


def _resolve_model(model: str | BaseChatModel | None) -> BaseChatModel:
    """Resolve model argument to a BaseChatModel instance."""
    if model is None:
        return init_chat_model(DEFAULT_MODEL)
    if isinstance(model, str):
        return init_chat_model(model)
    return model


def _make_rebuild_callback(
    *,
    resolver: SkillResolver | None,
    tools: Sequence[BaseAgentTool[Any]],
    summary_template: str,
    user_instructions: str | None,
) -> Callable[[str], Awaitable[list[BaseMessage]]]:
    """Create a callback that rebuilds messages after compaction.

    At compaction time, this callback:
    1. Re-discovers skills from filesystem (live snapshot, not stale cache)
    2. Composes a fresh system prompt using RESUMED_SESSION profile
    3. Returns [SystemMessage(new_prompt), HumanMessage(summary)]
    """

    async def rebuild(summary: str) -> list[BaseMessage]:
        # Import here to avoid circular import at module level
        from langchain_core.messages import HumanMessage, SystemMessage

        # Re-discover skills for live snapshot
        current_skills: list[Skill] = []
        if resolver is not None:
            current_skills = await resolver.discover()

        # Compose fresh system prompt with current capabilities
        ctx = AgentContext(
            tools=list(tools),
            skills=current_skills,
            mcps=[],
            user_instructions=user_instructions,
        )
        new_system_prompt = compose(RESUMED_SESSION, ctx)

        # Format summary into the rebuild template
        summary_content = substitute(summary_template, SUMMARY_CONTENT=summary)

        return [
            SystemMessage(content=new_system_prompt),
            HumanMessage(content=summary_content),
        ]

    return rebuild
