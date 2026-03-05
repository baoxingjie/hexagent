"""Local computer implementations.

This package provides computer implementations that run on the local machine.
"""

from openagent.computer.local.native import LocalNativeComputer
from openagent.computer.local.vm import LocalVMComputer

__all__ = ["LocalNativeComputer", "LocalVMComputer"]
