"""
Adapter Pool Controller for AgentBench

A load-balancing controller that manages multiple task_server_adapter instances
for high-concurrency scenarios with RL training frameworks like AReaL.

Architecture:
    AReaL → AdapterPoolController (5000) → Adapter (5001)
                                         → Adapter (5002)
                                         → Adapter (5003)
                                         → ...

Usage:
    python -m src.server.adapter_pool_controller alfworld-std \
    --config "configs/tasks/alfworld.yaml" \
    --port 5000 \
    --num-adapters 8 \
    --max-restarts 10 \
    --log-dir "adapter_logs" \
    > controller.log 2>&1
"""

import argparse
import asyncio
import os
import shlex
import subprocess
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime
from enum import Enum
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


def _build_docker_base_cmd(container_runtime: str) -> List[str]:
    """Build container runtime command prefix from environment overrides."""
    if container_runtime == "docker":
        cmd = [os.environ.get("AGENTBENCH_DOCKER_BIN", "docker")]
        extra_args = os.environ.get("AGENTBENCH_DOCKER_GLOBAL_ARGS", "")
        if extra_args:
            cmd.extend(shlex.split(extra_args))
        return cmd

    cmd = [os.environ.get("AGENTBENCH_PODMAN_BIN", "podman")]

    podman_root = os.environ.get("AGENTBENCH_PODMAN_ROOT")
    if podman_root:
        cmd.extend(["--root", podman_root])

    podman_runroot = os.environ.get("AGENTBENCH_PODMAN_RUNROOT")
    if podman_runroot:
        cmd.extend(["--runroot", podman_runroot])

    podman_tmpdir = os.environ.get("AGENTBENCH_PODMAN_TMPDIR")
    if podman_tmpdir:
        cmd.extend(["--tmpdir", podman_tmpdir])

    storage_opt = os.environ.get("AGENTBENCH_PODMAN_STORAGE_OPT")
    if storage_opt:
        cmd.extend(["--storage-opt", storage_opt])

    extra_args = os.environ.get("AGENTBENCH_PODMAN_GLOBAL_ARGS", "")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


# ============================================================================
# Request/Response Models (same as task_server_adapter)
# ============================================================================

class StartEpisodeRequest(BaseModel):
    sample_id: Union[str, int]
    config: Optional[Dict[str, Any]] = {}


class StepEpisodeRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]


class CancelEpisodeRequest(BaseModel):
    episode_id: str


class Observation(BaseModel):
    type: str
    content: Union[str, List[Dict[str, str]]]


class EpisodeResponse(BaseModel):
    episode_id: str
    observation: Optional[Observation] = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


# ============================================================================
# Load Balancing Strategies
# ============================================================================

class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round-robin"
    LEAST_BUSY = "least-busy"


# ============================================================================
# Adapter State Tracking
# ============================================================================

class AdapterStatus(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"   # hit episode limit; finishing in-flight episodes before restart
    DEAD = "dead"


@dataclass
class AdapterState:
    """Tracks state of a child adapter instance."""
    port: int
    process: Optional[subprocess.Popen]
    status: AdapterStatus = AdapterStatus.STARTING
    active_episodes: int = 0
    total_episodes: int = 0
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    restart_count: int = 0
    log_file: Optional[str] = None
    started_at: Optional[datetime] = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.url}/api"

    def is_process_alive(self) -> bool:
        """Check if the subprocess is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None


# ============================================================================
# Adapter Pool Controller
# ============================================================================

class AdapterPoolController:
    """
    Manages a pool of task_server_adapter instances and load-balances requests.
    """

    def __init__(
        self,
        task_name: str,
        config_path: str,
        base_port: int = 5000,
        num_adapters: int = 4,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        health_check_interval: int = 30,
        startup_timeout: int = 180,
        max_restarts: int = 5,
        log_dir: str = "adapter_logs",
        restart_after_episodes: int = 0,
        container_runtime: str = "docker",
    ):
        self.task_name = task_name
        self.config_path = config_path
        self.base_port = base_port
        self.num_adapters = num_adapters
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.startup_timeout = startup_timeout
        self.max_restarts = max_restarts
        self.log_dir = log_dir
        self.restart_after_episodes = restart_after_episodes
        self.container_runtime = container_runtime
        self.container_base_cmd = _build_docker_base_cmd(container_runtime)

        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # State tracking
        self.adapters: Dict[int, AdapterState] = {}  # port -> state
        self.episode_to_adapter: Dict[str, int] = {}  # episode_id -> port
        self.round_robin_index = 0

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._process_monitor_task: Optional[asyncio.Task] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Shutdown flag
        self._shutting_down = False

    async def initialize(self):
        """Start adapter pool and background tasks."""
        print(f"\n{'='*70}")
        print(f"Starting Adapter Pool Controller")
        print(f"{'='*70}")
        print(f"Task: {self.task_name}")
        print(f"Adapters: {self.num_adapters}")
        print(f"Strategy: {self.strategy.value}")
        print(f"Ports: {self.base_port + 1} - {self.base_port + self.num_adapters}")
        print(f"Container command: {shlex.join(self.container_base_cmd)}")
        print(f"{'='*70}\n")

        # Create HTTP session for proxying
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )

        # Launch adapter processes
        await self._launch_adapters()

        # Wait for adapters to become healthy
        await self._wait_for_adapters_ready()

        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._process_monitor_task = asyncio.create_task(self._process_monitor_loop())

        print(f"\n{'='*70}")
        print(f"Adapter Pool Ready!")
        print(f"Log directory: {os.path.abspath(self.log_dir)}")
        print(f"{'='*70}\n")

    async def shutdown(self):
        """Gracefully shutdown all adapters."""
        self._shutting_down = True
        print("\nShutting down adapter pool...")

        # Cancel background tasks
        for task in [self._health_check_task, self._process_monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Terminate adapter processes and their containers
        for port, adapter in self.adapters.items():
            print(f"  Stopping adapter on port {port}...")
            try:
                self._kill_adapter_process(adapter)
            except Exception as e:
                print(f"  Error stopping adapter {port}: {e}")

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()

        print("Adapter pool shutdown complete.")

    async def _launch_adapters(self):
        """Launch adapter subprocesses."""
        for i in range(self.num_adapters):
            port = self.base_port + 1 + i
            await self._launch_single_adapter(port)

    async def _launch_single_adapter(self, port: int, is_restart: bool = False, reset_restart_count: bool = False):
        """Launch a single adapter subprocess."""
        # Ensure no stale container is holding the port
        container = self._container_name(port)
        try:
            subprocess.run(
                self.container_base_cmd + ["rm", "-f", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        cmd = [
            sys.executable, "-m", "src.server.task_server_adapter",
            self.task_name,
            "--config", self.config_path,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--container-runtime", self.container_runtime,
        ]

        # Create log file for this adapter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"adapter_{port}_{timestamp}.log")

        restart_info = ""
        if is_restart and port in self.adapters:
            restart_info = f" (restart #{self.adapters[port].restart_count + 1})"

        print(f"Launching adapter on port {port}{restart_info}...")
        print(f"  Log file: {log_file}")

        # Open log file for writing
        log_handle = open(log_file, "w")
        launch_ts = datetime.now().isoformat()
        prior_restart_count = self.adapters[port].restart_count if port in self.adapters else 0
        launch_header = [
            "=" * 70,
            f"[adapter-launch] ts={launch_ts}",
            f"[adapter-launch] controller_pid={os.getpid()}",
            f"[adapter-launch] cwd={os.getcwd()}",
            f"[adapter-launch] python={sys.executable}",
            f"[adapter-launch] task_name={self.task_name}",
            f"[adapter-launch] config_path={self.config_path}",
            f"[adapter-launch] port={port}",
            f"[adapter-launch] is_restart={is_restart}",
            f"[adapter-launch] prior_restart_count={prior_restart_count}",
            f"[adapter-launch] reset_restart_count={reset_restart_count}",
            f"[adapter-launch] container_runtime={self.container_runtime}",
            f"[adapter-launch] cmd={shlex.join(cmd)}",
            "=" * 70,
            "",
        ]
        log_handle.write("\n".join(launch_header))
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # Own process group — killpg will reach the Docker child too
        )

        # Track restart count.  Proactive (episode-limit) restarts reset the
        # counter so they don't eat into the crash-loop budget.
        restart_count = 0
        if is_restart and port in self.adapters:
            restart_count = 0 if reset_restart_count else self.adapters[port].restart_count + 1

        self.adapters[port] = AdapterState(
            port=port,
            process=process,
            status=AdapterStatus.STARTING,
            restart_count=restart_count,
            log_file=log_file,
            started_at=datetime.now(),
        )

    async def _wait_for_adapters_ready(self):
        """Wait for all adapters to become healthy."""
        print(f"\nWaiting for {self.num_adapters} adapters to become ready...")

        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            healthy_count = 0

            for port, adapter in self.adapters.items():
                if adapter.status == AdapterStatus.HEALTHY:
                    healthy_count += 1
                    continue

                # Try health check
                try:
                    async with self._http_session.get(
                        f"{adapter.api_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            adapter.status = AdapterStatus.HEALTHY
                            adapter.last_health_check = datetime.now()
                            healthy_count += 1
                            print(f"  Adapter {port}: HEALTHY")
                except Exception:
                    pass

            if healthy_count == self.num_adapters:
                print(f"\nAll {self.num_adapters} adapters are healthy!")
                return

            await asyncio.sleep(2)

        # Check how many are healthy
        healthy = [p for p, a in self.adapters.items() if a.status == AdapterStatus.HEALTHY]
        if not healthy:
            raise RuntimeError("No adapters became healthy within timeout!")

        print(f"\nWarning: Only {len(healthy)}/{self.num_adapters} adapters are healthy")

    async def _health_check_loop(self):
        """Background task to monitor adapter health."""
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_adapters()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check error: {e}")

    async def _check_all_adapters(self):
        """Check health of all adapters."""
        for port, adapter in self.adapters.items():
            # First check if process is alive
            if not adapter.is_process_alive():
                adapter.status = AdapterStatus.DEAD
                adapter.consecutive_failures += 1
                print(f"Warning: Adapter {port} process is dead!")
                continue

            try:
                async with self._http_session.get(
                    f"{adapter.api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Don't overwrite DRAINING — that's an intentional
                        # state set by the restart logic; clearing it would
                        # let new episodes route here again.
                        if adapter.status != AdapterStatus.DRAINING:
                            adapter.status = AdapterStatus.HEALTHY
                        adapter.active_episodes = data.get("active_episodes", 0)
                        print(f"=======================================Adapter {adapter.port} active_episode: {adapter.active_episodes}")
                        adapter.last_health_check = datetime.now()
                        adapter.consecutive_failures = 0
                    else:
                        adapter.consecutive_failures += 1
            except Exception:
                adapter.consecutive_failures += 1

            if adapter.consecutive_failures >= 3:
                adapter.status = AdapterStatus.UNHEALTHY
                print(f"Warning: Adapter {port} is unhealthy (consecutive failures: {adapter.consecutive_failures})")

    async def _process_monitor_loop(self):
        """Background task to monitor adapter processes and restart dead ones."""
        while not self._shutting_down:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._check_and_restart_adapters()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Process monitor error: {e}")

    async def _check_and_restart_adapters(self):
        """Check for dead/unhealthy adapters and restart them."""
        for port, adapter in list(self.adapters.items()):
            should_restart = False
            reason = ""

            # Check if process died
            if not adapter.is_process_alive():
                should_restart = True
                reason = "process died"
                # Try to get exit code
                if adapter.process:
                    exit_code = adapter.process.poll()
                    reason = f"process exited with code {exit_code}"

            # Check if too many consecutive health check failures
            elif adapter.consecutive_failures >= 5:
                should_restart = True
                reason = f"consecutive health failures ({adapter.consecutive_failures})"

            # Already draining — restart now that the last episode finished
            elif adapter.status == AdapterStatus.DRAINING and adapter.active_episodes == 0:
                # Check if we have enough healthy adapters before restarting
                healthy_count = len([a for a in self.adapters.values() if a.status == AdapterStatus.HEALTHY])
                healthy_percentage = healthy_count / self.num_adapters

                if healthy_percentage < 0.5 + 0.01:
                    # Not enough healthy adapters - bring this adapter back to service
                    # instead of restarting it
                    print(f"  Adapter {port}: canceling restart (only {healthy_count}/{self.num_adapters} healthy < 50%)")
                    print(f"  Adapter {port}: returning to HEALTHY state to handle new requests")
                    adapter.status = AdapterStatus.HEALTHY
                    adapter.total_episodes = 0  # Reset counter to give it a fresh lifecycle
                    continue

                should_restart = True
                reason = f"drain complete (total {adapter.total_episodes}/{self.restart_after_episodes})"

            # Proactive restart: container has handled enough episodes that
            # /tmp bloat (e.g. fast_downward temp dirs) warrants a fresh start.
            elif (self.restart_after_episodes > 0
                  and adapter.total_episodes >= self.restart_after_episodes):
                if adapter.active_episodes == 0:
                    # Check if we have enough healthy adapters before restarting
                    healthy_count = len([a for a in self.adapters.values() if a.status == AdapterStatus.HEALTHY])
                    healthy_percentage = healthy_count / self.num_adapters

                    if healthy_percentage < 0.5 + 0.01:
                        # Not enough healthy adapters - skip restart for now
                        print(f"  Adapter {port}: deferring restart (only {healthy_count}/{self.num_adapters} healthy < 50%)")
                        # Keep it in current state, will retry on next check
                        continue

                    # Nothing in flight — restart immediately
                    should_restart = True
                    reason = f"episode limit reached ({adapter.total_episodes}/{self.restart_after_episodes})"
                elif adapter.status != AdapterStatus.DRAINING:
                    # Episodes still running — stop routing new work here and
                    # wait for them to finish before recycling the container.
                    print(f"Adapter {port}: {adapter.status}")
                    if adapter.status == AdapterStatus.HEALTHY:
                        healthy_count = len([a for a in self.adapters.values() if a.status == AdapterStatus.HEALTHY])
                        print(f"healthy_count: {healthy_count} / {self.num_adapters}")
                        healthy_percentage = healthy_count / self.num_adapters

                        if healthy_percentage < 0.5 + 0.01:
                            # Not enough healthy adapters - bring this adapter back to service
                            # instead of restarting it
                            print(f"  Adapter {port}: canceling restart (only {healthy_count}/{self.num_adapters} healthy < 50%)")
                            print(f"  Adapter {port}: returning to HEALTHY state to handle new requests")
                            adapter.status = AdapterStatus.HEALTHY
                            adapter.total_episodes = 0  # Reset counter to give it a fresh lifecycle
                            continue
                    adapter.status = AdapterStatus.DRAINING
                    print(f"  Adapter {port}: draining ({adapter.active_episodes} active, "
                          f"{adapter.total_episodes}/{self.restart_after_episodes} total)")

            if should_restart:
                is_proactive = reason.startswith("episode limit") or reason.startswith("drain complete")

                # Proactive restarts are planned recycling, not crash recovery —
                # they bypass the max_restarts guard (and reset its counter).
                if not is_proactive and adapter.restart_count >= self.max_restarts:
                    print(f"ERROR: Adapter {port} has exceeded max restarts ({self.max_restarts}). Not restarting.")
                    adapter.status = AdapterStatus.DEAD
                    continue

                print(f"\n{'='*50}")
                print(f"Restarting adapter {port}: {reason}")
                print(f"  Log file: {adapter.log_file}")
                print(f"{'='*50}")

                # Cleanup episodes owned by this adapter
                await self._cleanup_adapter_episodes(port)

                # Kill the old process and its container via process group
                self._kill_adapter_process(adapter)

                # Launch new adapter (proactive restarts reset crash counter)
                await self._launch_single_adapter(port, is_restart=True, reset_restart_count=is_proactive)

                # Wait for it to become healthy
                await self._wait_for_single_adapter(port, timeout=60)

    async def _wait_for_single_adapter(self, port: int, timeout: int = 60):
        """Wait for a single adapter to become healthy."""
        adapter = self.adapters[port]
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not adapter.is_process_alive():
                print(f"  Adapter {port} process died during startup!")
                return False

            try:
                async with self._http_session.get(
                    f"{adapter.api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        adapter.status = AdapterStatus.HEALTHY
                        adapter.last_health_check = datetime.now()
                        adapter.consecutive_failures = 0
                        print(f"  Adapter {port}: HEALTHY (restarted)")
                        return True
            except Exception:
                pass

            await asyncio.sleep(2)

        print(f"  Adapter {port} failed to become healthy within {timeout}s")
        return False

    async def _cleanup_adapter_episodes(self, port: int):
        """Remove episode mappings for a dead adapter."""
        episodes_to_remove = [
            ep_id for ep_id, ep_port in self.episode_to_adapter.items()
            if ep_port == port
        ]

        if episodes_to_remove:
            print(f"  Cleaning up {len(episodes_to_remove)} orphaned episodes from adapter {port}")
            for ep_id in episodes_to_remove:
                del self.episode_to_adapter[ep_id]

    @staticmethod
    def _container_name(port: int) -> str:
        return f"agentbench-adapter-{port}"

    def _kill_adapter_process(self, adapter: AdapterState):
        """Kill the adapter subprocess and force-remove its container.

        Relies on ``<runtime> rm -f`` to guarantee the container (and its port
        mapping) is gone before we return.  The Python wrapper process is
        killed via process-group signal as a best-effort first step.
        """
        # 1. Best-effort: kill the process group (wrapper + runtime CLI)
        if adapter.process and adapter.is_process_alive():
            try:
                pgid = os.getpgid(adapter.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                adapter.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(adapter.process.pid), signal.SIGKILL)
                except OSError:
                    pass
                try:
                    adapter.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            except OSError:
                pass  # process group already gone

        # 2. Force-remove the container by name — this is the
        #    authoritative kill.  Even if the wrapper died without
        #    forwarding the signal, this will stop and remove the container.
        container = self._container_name(adapter.port)
        try:
            subprocess.run(
                self.container_base_cmd + ["rm", "-f", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # runtime missing or timed out — non-container task

    def _select_adapter(self) -> AdapterState:
        """Select an adapter based on the load balancing strategy."""
        healthy_adapters = [
            a for a in self.adapters.values()
            if a.status == AdapterStatus.HEALTHY
        ]

        if not healthy_adapters:
            raise HTTPException(
                status_code=503,
                detail="No healthy adapters available"
            )

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            # Round-robin selection
            adapter = healthy_adapters[self.round_robin_index % len(healthy_adapters)]
            self.round_robin_index += 1
            return adapter

        elif self.strategy == LoadBalanceStrategy.LEAST_BUSY:
            # Select adapter with fewest active episodes
            return min(healthy_adapters, key=lambda a: a.active_episodes)

        else:
            return healthy_adapters[0]

    def _get_adapter_for_episode(self, episode_id: str) -> AdapterState:
        """Get the adapter handling a specific episode."""
        if episode_id not in self.episode_to_adapter:
            raise HTTPException(
                status_code=404,
                detail=f"Episode {episode_id} not found"
            )

        port = self.episode_to_adapter[episode_id]
        return self.adapters[port]

    # ========================================================================
    # API Proxy Methods
    # ========================================================================

    async def get_task_info(self) -> Dict[str, Any]:
        """Get task info from any healthy adapter."""
        adapter = self._select_adapter()

        async with self._http_session.get(f"{adapter.api_url}/task/info") as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())
            data = await resp.json()

            # Add pool info
            data["pool_info"] = {
                "num_adapters": self.num_adapters,
                "healthy_adapters": len([a for a in self.adapters.values() if a.status == AdapterStatus.HEALTHY]),
                "total_active_episodes": sum(a.active_episodes for a in self.adapters.values()),
                "strategy": self.strategy.value,
            }
            return data

    async def start_episode(
        self,
        sample_id: Union[str, int],
        config: Dict[str, Any]
    ) -> EpisodeResponse:
        """Start episode on least-busy adapter."""
        adapter = self._select_adapter()

        request_data = {"sample_id": sample_id, "config": config}
        print(f"Staring sample {sample_id} by adapter {adapter.api_url}:{adapter.port}")

        async with self._http_session.post(
            f"{adapter.api_url}/episode/start",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                try:
                    error_detail = await resp.json()
                except Exception:
                    error_detail = await resp.text()
                print(
                    f"Adapter start_episode failed: adapter_port={adapter.port}, "
                    f"sample_id={sample_id}, status={resp.status}, detail={error_detail}"
                )
                raise HTTPException(
                    status_code=resp.status,
                    detail={
                        "upstream_status": resp.status,
                        "adapter_port": adapter.port,
                        "sample_id": sample_id,
                        "upstream_detail": error_detail,
                    },
                )

            data = await resp.json()
            episode_id = data["episode_id"]

            # Track which adapter owns this episode
            self.episode_to_adapter[episode_id] = adapter.port
            adapter.active_episodes += 1
            adapter.total_episodes += 1

            return EpisodeResponse(**data)

    async def step_episode(
        self,
        episode_id: str,
        action: Dict[str, Any]
    ) -> EpisodeResponse:
        """Route step to correct adapter."""
        adapter = self._get_adapter_for_episode(episode_id)

        request_data = {"episode_id": episode_id, "action": action}
        print(f"Stepping episode {episode_id} by adapter {adapter.api_url}")

        async with self._http_session.post(
            f"{adapter.api_url}/episode/step",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                try:
                    error_detail = await resp.json()
                except Exception:
                    error_detail = await resp.text()
                print(
                    f"Adapter step_episode failed: adapter_port={adapter.port}, "
                    f"episode_id={episode_id}, status={resp.status}, detail={error_detail}"
                )
                raise HTTPException(
                    status_code=resp.status,
                    detail={
                        "upstream_status": resp.status,
                        "adapter_port": adapter.port,
                        "episode_id": episode_id,
                        "upstream_detail": error_detail,
                    },
                )

            data = await resp.json()

            # Cleanup if episode is done
            if data.get("done", False):
                del self.episode_to_adapter[episode_id]
                adapter.active_episodes = max(0, adapter.active_episodes - 1)

            return EpisodeResponse(**data)

    async def cancel_episode(self, episode_id: str) -> Dict[str, str]:
        """Route cancel to correct adapter."""
        adapter = self._get_adapter_for_episode(episode_id)

        request_data = {"episode_id": episode_id}

        async with self._http_session.post(
            f"{adapter.api_url}/episode/cancel",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                try:
                    error_detail = await resp.json()
                except Exception:
                    error_detail = await resp.text()
                print(
                    f"Adapter cancel_episode failed: adapter_port={adapter.port}, "
                    f"episode_id={episode_id}, status={resp.status}, detail={error_detail}"
                )
                raise HTTPException(
                    status_code=resp.status,
                    detail={
                        "upstream_status": resp.status,
                        "adapter_port": adapter.port,
                        "episode_id": episode_id,
                        "upstream_detail": error_detail,
                    },
                )

            # Cleanup tracking
            del self.episode_to_adapter[episode_id]
            adapter.active_episodes = max(0, adapter.active_episodes - 1)

            return await resp.json()

    async def get_health(self) -> Dict[str, Any]:
        """Get pool health status."""
        # adapter_statuses = {}
        # for port, adapter in self.adapters.items():
        #     adapter_statuses[port] = {
        #         "status": adapter.status.value,
        #         "process_alive": adapter.is_process_alive(),
        #         "active_episodes": adapter.active_episodes,
        #         "total_episodes": adapter.total_episodes,
        #         "consecutive_failures": adapter.consecutive_failures,
        #         "restart_count": adapter.restart_count,
        #         "log_file": adapter.log_file,
        #         "started_at": adapter.started_at.isoformat() if adapter.started_at else None,
        #         "last_health_check": adapter.last_health_check.isoformat() if adapter.last_health_check else None,
        #     }

        healthy_count = len([a for a in self.adapters.values() if a.status == AdapterStatus.HEALTHY])
        draining_count = len([a for a in self.adapters.values() if a.status == AdapterStatus.DRAINING])
        unhealthy_count = len([a for a in self.adapters.values() if a.status == AdapterStatus.UNHEALTHY])
        dead_count = len([a for a in self.adapters.values() if a.status == AdapterStatus.DEAD])
        starting_cound = len([a for a in self.adapters.values() if a.status == AdapterStatus.STARTING])
        total_restarts = sum(a.restart_count for a in self.adapters.values())

        adapter_active_episodes = [a.active_episodes for a in self.adapters.values()]
        adapter_active_episodes = " ".join(map(str, adapter_active_episodes))
        adapter_total_episodes = [a.total_episodes for a in self.adapters.values()]
        adapter_total_episodes = " ".join(map(str, adapter_total_episodes))

        return {
            "status": "healthy" if healthy_count > 0 else "unhealthy",
            "task_name": self.task_name,
            "num_adapters": self.num_adapters,
            "healthy_count": healthy_count,
            "draining_count": draining_count,
            "unhealthy_count": unhealthy_count,
            "dead_count": dead_count,
            "starting_cound": starting_cound,
            "total_active_episodes": len(self.episode_to_adapter),
            "adapter_active_episodes": adapter_active_episodes,
            "adapter_total_episodes": adapter_total_episodes,
            "total_restarts": total_restarts,
            "strategy": self.strategy.value,
            "log_dir": os.path.abspath(self.log_dir),
            # "adapters": adapter_statuses,
        }


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app(controller: AdapterPoolController) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="AgentBench Adapter Pool Controller",
        description="Load-balancing controller for multiple task_server_adapter instances",
        version="1.0.0"
    )

    @app.on_event("startup")
    async def startup():
        await controller.initialize()

    @app.on_event("shutdown")
    async def shutdown():
        await controller.shutdown()

    @app.get("/api/task/info")
    async def get_task_info():
        return await controller.get_task_info()

    @app.post("/api/episode/start", response_model=EpisodeResponse)
    async def start_episode(request: StartEpisodeRequest):
        return await controller.start_episode(request.sample_id, request.config)

    @app.post("/api/episode/step", response_model=EpisodeResponse)
    async def step_episode(request: StepEpisodeRequest):
        return await controller.step_episode(request.episode_id, request.action)

    @app.post("/api/episode/cancel")
    async def cancel_episode(request: CancelEpisodeRequest):
        return await controller.cancel_episode(request.episode_id)

    @app.get("/api/health")
    async def health_check():
        # return await controller.get_health()
        import json
        data = await controller.get_health()
        return Response(content=json.dumps(data, indent=2) + "\n",
                        media_type="application/json")

    @app.get("/api/pool/status")
    async def pool_status():
        """Detailed pool status endpoint."""
        # return await controller.get_health()
        import json
        data = await controller.get_health()
        return Response(content=json.dumps(data, indent=2) + "\n",
                        media_type="application/json")

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run a pool of AgentBench task server adapters with load balancing"
    )
    parser.add_argument(
        "task_name",
        type=str,
        help="Task name from config (e.g., 'alfworld-std')"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/tasks/alfworld.yaml",
        help="Path to task config file"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port for the controller (adapters use port+1, port+2, ...)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--num-adapters", "-n",
        type=int,
        default=4,
        help="Number of adapter instances to launch"
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["round-robin", "least-busy"],
        default="least-busy",
        help="Load balancing strategy"
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=120,
        help="Timeout in seconds to wait for adapters to start"
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=5,
        help="Maximum number of restarts per adapter before giving up"
    )
    parser.add_argument(
        "--restart-after-episodes", "-e",
        type=int,
        default=0,
        help="Proactively restart each adapter after this many episodes (0 = disabled). "
             "Prevents /tmp bloat from long-lived containers (e.g. fast_downward temp dirs)."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="adapter_logs",
        help="Directory to store adapter log files"
    )
    parser.add_argument(
        "--container-runtime",
        type=str,
        choices=["docker", "podman"],
        default="podman",
        help="Container runtime used by child adapters for containerized tasks"
    )

    args = parser.parse_args()

    # Create controller
    controller = AdapterPoolController(
        task_name=args.task_name,
        config_path=args.config,
        base_port=args.port,
        num_adapters=args.num_adapters,
        strategy=LoadBalanceStrategy(args.strategy),
        startup_timeout=args.startup_timeout,
        max_restarts=args.max_restarts,
        log_dir=args.log_dir,
        restart_after_episodes=args.restart_after_episodes,
        container_runtime=args.container_runtime,
    )

    # Create app
    app = create_app(controller)

    print(f"\n{'='*70}")
    print(f"Adapter Pool Controller")
    print(f"{'='*70}")
    print(f"Controller URL: http://{args.host}:{args.port}")
    print(f"API Base:       http://{args.host}:{args.port}/api")
    print(f"Health:         http://{args.host}:{args.port}/api/health")
    print(f"Pool Status:    http://{args.host}:{args.port}/api/pool/status")
    print(f"Log directory:  {os.path.abspath(args.log_dir)}")
    print(f"Runtime:        {args.container_runtime}")
    print(f"Max restarts:   {args.max_restarts}")
    print(f"Episode limit:  {args.restart_after_episodes or 'disabled'}")
    print(f"{'='*70}\n")

    # Let uvicorn handle SIGINT/SIGTERM natively so that its graceful
    # shutdown triggers FastAPI's "shutdown" event → controller.shutdown()
    # → _kill_adapter_process (with runtime rm -f) for every adapter.
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
