"""Harbor custom environment backed by rootless Podman.

This backend is intentionally small and targets the Harbor CodeContests layout
used by the AstraFlow t-bench RL recipe: one Linux Dockerfile per task and no
Docker Compose overlays. It is loaded by Harbor via
``--environment-import-path examples.terminal-bench.harbor_podman_env:PodmanEnvironment``.
"""

from __future__ import annotations

import asyncio
import os
import pwd
import re
import shlex
import shutil
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.task.config import TaskOS


def _podman_bin() -> str:
    return os.environ.get("HARBOR_PODMAN_BIN", "podman")


def _user_has_subid_mapping(path: str, username: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(":")
                if len(parts) >= 3 and parts[0] == username and int(parts[2]) > 1:
                    return True
    except (OSError, ValueError):
        return False
    return False


def _ignore_chown_errors_enabled() -> bool:
    override = os.environ.get("HARBOR_PODMAN_IGNORE_CHOWN_ERRORS")
    if override is not None:
        return override.lower() not in {"", "0", "false", "no", "off"}
    try:
        username = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        username = os.environ.get("USER", "")
    return not (
        username
        and _user_has_subid_mapping("/etc/subuid", username)
        and _user_has_subid_mapping("/etc/subgid", username)
    )


def _podman_cmd() -> list[str]:
    cmd = [_podman_bin()]
    if _ignore_chown_errors_enabled():
        cmd.extend(["--storage-opt", "ignore_chown_errors=true"])
    return cmd


def _podman_network_mode() -> str | None:
    override = os.environ.get("HARBOR_PODMAN_NETWORK")
    if override is not None:
        return override or None
    if shutil.which("pasta") is None and shutil.which("slirp4netns"):
        return "slirp4netns"
    return None

def _sanitize_image_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9._-]", "-", name)


def _sanitize_container_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_.-]", "-", name)[:250]


class PodmanEnvironment(BaseEnvironment):
    """Dockerfile-only Harbor environment implemented with ``podman``."""

    _image_build_locks: dict[str, asyncio.Lock] = {}

    def __init__(self, *args, **kwargs):
        self._container_name: str | None = None
        self._image_name: str | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "podman"

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which(_podman_bin()):
            raise SystemExit(
                "Podman is not installed or not on PATH. Install podman in the "
                "Harbor environment before using PodmanEnvironment."
            )
        try:
            subprocess = asyncio.run(cls._preflight_info())
        except Exception as exc:  # pragma: no cover - defensive CLI error path
            raise SystemExit(f"Podman preflight failed: {exc}") from exc
        if subprocess.return_code != 0:
            output = subprocess.stderr or subprocess.stdout or "no output"
            raise SystemExit(f"Podman is not usable: {output}")

    @staticmethod
    async def _preflight_info() -> ExecResult:
        return await PodmanEnvironment._run_host([*_podman_cmd(), "info"], check=False, timeout_sec=20)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            mounted=True,
            docker_compose=False,
            windows=False,
            gpus=False,
        )

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    def _validate_definition(self) -> None:
        if self.task_env_config.os == TaskOS.WINDOWS:
            raise RuntimeError("PodmanEnvironment only supports Linux tasks.")
        if self._compose_path.exists() or self.extra_docker_compose_paths:
            raise RuntimeError(
                "PodmanEnvironment only supports Dockerfile-only Harbor tasks; "
                "docker-compose.yaml and --extra-docker-compose are unsupported."
            )
        if not self.task_env_config.docker_image and not self._dockerfile_path.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {self._dockerfile_path}")

    @staticmethod
    async def _run_host(
        args: list[str],
        *,
        check: bool = True,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except asyncio.TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds: {shlex.join(args)}")

        result = ExecResult(
            stdout=stdout_bytes.decode(errors="replace") if stdout_bytes else None,
            stderr=stderr_bytes.decode(errors="replace") if stderr_bytes else None,
            return_code=process.returncode or 0,
        )
        if check and result.return_code != 0:
            output = result.stderr or result.stdout or "no output"
            raise RuntimeError(
                f"Podman command failed (rc={result.return_code}): "
                f"{shlex.join(args)}\n{output}"
            )
        return result

    async def _image_exists(self, image_name: str) -> bool:
        result = await self._run_host([*_podman_cmd(), "image", "exists", image_name], check=False)
        return result.return_code == 0

    async def _ensure_image(self, force_build: bool) -> str:
        if self.task_env_config.docker_image:
            image_name = self.task_env_config.docker_image
            if force_build or not await self._image_exists(image_name):
                await self._run_host([*_podman_cmd(), "pull", image_name])
            return image_name

        image_name = f"hb__{_sanitize_image_name(self.environment_name)}"
        lock = self._image_build_locks.setdefault(image_name, asyncio.Lock())
        async with lock:
            await self._run_host(
                [
                    *_podman_cmd(),
                    "build",
                    *(["--network", _podman_network_mode()] if _podman_network_mode() else []),
                    "--tag",
                    image_name,
                    "--file",
                    str(self._dockerfile_path.resolve()),
                    str(self.environment_dir.resolve()),
                ]
            )
        return image_name

    def _volume_args(self) -> list[str]:
        args: list[str] = []
        for mount in self._mounts:
            if mount.get("type") != "bind":
                raise RuntimeError(
                    "PodmanEnvironment currently supports bind mounts only; "
                    f"got mount type {mount.get('type')!r}."
                )
            source = Path(str(mount["source"])).expanduser()
            if mount.get("bind", {}).get("create_host_path") is not False:
                source.mkdir(parents=True, exist_ok=True)
            mode = "ro" if mount.get("read_only") else "rw"
            args.extend(["--volume", f"{source.resolve()}:{mount['target']}:{mode}"])
        return args

    def _resource_args(self) -> list[str]:
        args: list[str] = []
        if self._effective_cpus:
            args.extend(["--cpus", str(self._effective_cpus)])
        if self._effective_memory_mb:
            args.extend(["--memory", f"{self._effective_memory_mb}m"])
        return args

    async def start(self, force_build: bool) -> None:
        image_name = await self._ensure_image(force_build=force_build)
        container_name = _sanitize_container_name(f"harbor-{self.session_id}")
        self._image_name = image_name
        self._container_name = container_name

        await self._run_host([*_podman_cmd(), "rm", "--force", container_name], check=False)

        cmd = [
            *_podman_cmd(),
            "run",
            "--detach",
            "--name",
            container_name,
            *self._volume_args(),
            *self._resource_args(),
        ]
        if not self.task_env_config.allow_internet:
            cmd.extend(["--network", "none"])
        elif _podman_network_mode():
            cmd.extend(["--network", _podman_network_mode()])
        for key, value in self._persistent_env.items():
            cmd.extend(["--env", f"{key}={value}"])
        cmd.extend([image_name, "sh", "-c", "sleep infinity"])
        await self._run_host(cmd)
        await self.ensure_dirs(self._mount_targets(writable_only=True))

    async def stop(self, delete: bool):
        if not self._container_name:
            return
        try:
            await self.prepare_logs_for_host()
            if delete:
                await self._run_host([*_podman_cmd(), "rm", "--force", self._container_name], check=False)
            else:
                await self._run_host([*_podman_cmd(), "stop", self._container_name], check=False)
        finally:
            self._container_name = None

    async def prepare_logs_for_host(self) -> None:
        try:
            for target in self._mount_targets(writable_only=True):
                await self._chown_to_host_user(target, recursive=True)
        except Exception as exc:
            self.logger.warning(f"Failed to chown mounted Harbor paths: {exc}")

    async def _chown_to_host_user(self, path: str, recursive: bool = False) -> None:
        flag = "-R " if recursive else ""
        await self.exec(f"chown {flag}{os.getuid()}:{os.getgid()} {shlex.quote(path)}", user="root")

    async def upload_file(self, source_path: Path | str, target_path: str):
        if not self._container_name:
            raise RuntimeError("Podman container is not started")
        await self._run_host([*_podman_cmd(), "cp", str(source_path), f"{self._container_name}:{target_path}"])

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        if not self._container_name:
            raise RuntimeError("Podman container is not started")
        await self.ensure_dirs([target_dir], chmod=False)
        await self._run_host([*_podman_cmd(), "cp", f"{source_dir}/.", f"{self._container_name}:{target_dir}"])

    async def download_file(self, source_path: str, target_path: Path | str):
        if not self._container_name:
            raise RuntimeError("Podman container is not started")
        await self._chown_to_host_user(source_path)
        await self._run_host([*_podman_cmd(), "cp", f"{self._container_name}:{source_path}", str(target_path)])

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        if not self._container_name:
            raise RuntimeError("Podman container is not started")
        await self._chown_to_host_user(source_dir, recursive=True)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        await self._run_host([*_podman_cmd(), "cp", f"{self._container_name}:{source_dir}/.", str(target_dir)])

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._container_name:
            raise RuntimeError("Podman container is not started")

        cmd = [*_podman_cmd(), "exec"]
        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            cmd.extend(["--user", str(resolved_user)])
        if cwd is not None:
            cmd.extend(["--workdir", cwd])
        merged_env = self._merge_env(env)
        if merged_env:
            for key, value in merged_env.items():
                cmd.extend(["--env", f"{key}={value}"])
        cmd.extend([self._container_name, "bash", "-c", command])
        return await self._run_host(cmd, check=False, timeout_sec=timeout_sec)
