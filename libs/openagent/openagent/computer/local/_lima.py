"""Lima VM instance management via limactl.

Manages the lifecycle of a Lima VM (create, start, stop, delete, shell).
This is infrastructure — not a Computer implementation. Session management
lives in ``LocalVMComputer``.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openagent.computer.base import ExecutionMetadata

if TYPE_CHECKING:
    from pathlib import Path

from openagent.exceptions import LimaError, MissingDependencyError, UnsupportedPlatformError
from openagent.types import CLIResult


@dataclass(frozen=True)
class Mount:
    """A host-to-guest filesystem mount.

    Attributes:
        location: Host path to mount.
        mount_point: Guest path (e.g. ``/sessions/{name}/mnt/{basename}``).
        writable: Whether the mount is writable.
    """

    location: str
    mount_point: str
    writable: bool = True


class LimaVM:
    """Manages a single Lima VM instance.

    Parameters:
        instance: Lima instance name.
    """

    def __init__(self, instance: str) -> None:
        if sys.platform != "darwin":
            msg = f"Lima is a macOS hypervisor — it cannot run on {sys.platform}"
            raise UnsupportedPlatformError(msg)
        if not shutil.which("limactl"):
            msg = "limactl not found. Install Lima: https://lima-vm.io"
            raise MissingDependencyError(msg)
        self._instance = instance

    @property
    def instance(self) -> str:
        """Read-only Lima instance name."""
        return self._instance

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status(self) -> str | None:
        """Return the instance status string, or ``None`` if it doesn't exist."""
        proc = await asyncio.create_subprocess_exec(
            "limactl",
            "list",
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await proc.communicate()

        if proc.returncode != 0:
            return None

        for line in stdout_bytes.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if entry.get("name") == self._instance:
                status: str | None = entry.get("status")
                return status

        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def build(self, yaml_path: Path | str) -> None:
        """Create and provision a new VM instance from a YAML config.

        Raises ``LimaError`` if the instance already exists.
        """
        current = await self.status()
        if current is not None:
            msg = f"Lima instance '{self._instance}' already exists (status: {current})"
            raise LimaError(msg)

        cmd = [
            "limactl",
            "start",
            f"--name={self._instance}",
            str(yaml_path),
            "--tty=false",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._poll_until_running()
        except BaseException:
            proc.kill()
            await proc.wait()
            raise
        proc.terminate()
        await proc.wait()
        await self._poll_until_shell_ready()

    async def start(self, *, mounts: list[Mount] | None = None) -> None:
        """Start the VM instance, optionally adding mounts.

        - If the instance doesn't exist, raises ``LimaError``.
        - If already running with no new mounts, this is a no-op.
        - If already running with mounts, stops first then restarts with ``--set``.
        """
        current = await self.status()

        if current is None:
            msg = f"Lima instance '{self._instance}' does not exist. Create it first with build()."
            raise LimaError(msg)

        if current == "Running" and not mounts:
            return

        if current == "Running" and mounts:
            await self.stop()

        # Build the start command
        cmd = ["limactl", "start", self._instance, "--tty=false"]

        if mounts:
            mount_set = self._build_mount_set_arg(mounts)
            cmd.extend(["--set", mount_set])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._poll_until_running()
        except BaseException:
            proc.kill()
            await proc.wait()
            raise
        proc.terminate()
        await proc.wait()
        await self._poll_until_shell_ready()

    async def stop(self) -> None:
        """Stop the VM instance. No-op if already stopped."""
        current = await self.status()
        if current is None or current != "Running":
            return

        proc = await asyncio.create_subprocess_exec(
            "limactl",
            "stop",
            self._instance,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def delete(self) -> None:
        """Delete the VM instance forcefully."""
        proc = await asyncio.create_subprocess_exec(
            "limactl",
            "delete",
            "--force",
            self._instance,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    # ------------------------------------------------------------------
    # Shell execution
    # ------------------------------------------------------------------

    async def shell(
        self,
        command: str,
        *,
        user: str | None = None,
        cwd: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> CLIResult:
        """Execute a command inside the VM via ``limactl shell``.

        Parameters:
            command: Shell command to run.
            user: If set, run as this Linux user via ``sudo -u`` with login shell.
            cwd: Working directory inside the VM. For session users without
                explicit cwd, defaults to ``cd`` (home directory).
            timeout: Timeout in **seconds**. ``None`` means wait indefinitely.

        Returns:
            CLIResult with stdout, stderr, exit_code, and metadata.

        Raises:
            LimaError: On timeout or subprocess failure.
        """
        if user is not None:
            escaped = command.replace("'", r"'\''")
            cd_part = f"cd {shlex.quote(cwd)}" if cwd is not None else "cd"
            inner = f"sudo -u {shlex.quote(user)} -H bash -l -c '{cd_part} && {escaped}'"
        elif cwd is not None:
            inner = f"cd {shlex.quote(cwd)} && {command}"
        else:
            inner = command

        exec_args = [
            "limactl",
            "shell",
            "--workdir",
            "/",
            self._instance,
            "bash",
            "-c",
            inner,
        ]

        start_time = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            *exec_args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            msg = f"timed out after {timeout}s"
            raise LimaError(msg) from None

        stdout = stdout_bytes.decode("utf-8", errors="replace").removesuffix("\n")
        stderr = stderr_bytes.decode("utf-8", errors="replace").removesuffix("\n")

        # returncode is always set after communicate()
        rc: int = process.returncode if process.returncode is not None else -1
        return CLIResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=rc,
            metadata=ExecutionMetadata(duration_ms=int((time.monotonic() - start_time) * 1000)),
        )

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    async def copy(self, src: str, dst: str, *, host_to_guest: bool) -> None:
        """Copy a file between host and guest via ``limactl copy``.

        Args:
            src: Source path (host path if host_to_guest, else guest path).
            dst: Destination path (guest path if host_to_guest, else host path).
            host_to_guest: Direction of the copy.

        Raises:
            LimaError: If the copy fails.
        """
        if host_to_guest:
            copy_src = src
            copy_dst = f"{self._instance}:{dst}"
        else:
            copy_src = f"{self._instance}:{src}"
            copy_dst = dst

        proc = await asyncio.create_subprocess_exec(
            "limactl",
            "copy",
            copy_src,
            copy_dst,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate()

        if proc.returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            direction = "host→guest" if host_to_guest else "guest→host"
            msg = f"limactl copy failed ({direction}): {stderr}"
            raise LimaError(msg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _poll_until_running(self, timeout: float = 120, interval: float = 2) -> None:  # noqa: ASYNC109
        """Poll ``status()`` until the VM reports "Running"."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(interval)
            if await self.status() == "Running":
                return
        msg = f"Lima instance '{self._instance}' did not reach Running state within {timeout}s"
        raise LimaError(msg)

    async def _poll_until_shell_ready(self, timeout: float = 30, interval: float = 0.2) -> None:  # noqa: ASYNC109
        """Poll until ``limactl shell`` is usable.

        The VM status may report "Running" before the SSH/socket layer
        is fully initialised.  This method repeatedly attempts a trivial
        shell command and returns only once it succeeds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            proc = await asyncio.create_subprocess_exec(
                "limactl",
                "shell",
                "--workdir",
                "/",
                self._instance,
                "bash",
                "-c",
                "true",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()
            if rc == 0:
                return
            await asyncio.sleep(interval)
        msg = f"Lima instance '{self._instance}' shell did not become ready within {timeout}s"
        raise LimaError(msg)

    @staticmethod
    def _build_mount_set_arg(mounts: list[Mount]) -> str:
        """Build the ``--set`` value for Lima mount configuration."""
        entries = [{"location": m.location, "mountPoint": m.mount_point, "writable": m.writable} for m in mounts]
        return f".mounts += {json.dumps(entries)}"
