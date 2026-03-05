"""Framework-agnostic task tools.

These tools operate against a :class:`~openagent.tasks.TaskRegistry`
and are assembled by the agent factory.
"""

from openagent.tools.task.agent import AgentTool
from openagent.tools.task.output import TaskOutputTool
from openagent.tools.task.stop import TaskStopTool

__all__ = [
    "AgentTool",
    "TaskOutputTool",
    "TaskStopTool",
]
