"""Computer abstractions for OpenAgent."""

from openagent.computer.base import Computer, ExecutionMetadata
from openagent.computer.local.native import LocalNativeComputer
from openagent.computer.local.vm import LocalVMComputer
from openagent.computer.remote.e2b import RemoteE2BComputer

__all__ = [
    "Computer",
    "ExecutionMetadata",
    "LocalNativeComputer",
    "LocalVMComputer",
    "RemoteE2BComputer",
]
