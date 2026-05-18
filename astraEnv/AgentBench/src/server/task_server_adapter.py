"""
Task Server Adapter for AgentBench

This adapter wraps AgentBench tasks with the standard Task Server API,
making them compatible with AReaL's TaskServerWorkflow.

Usage:
    python -m src.server.task_server_adapter os-dev --port 5000
    python -m src.server.task_server_adapter os-dev --port 5000 > std.log 2>&1 &

    python -m src.server.task_server_adapter alfworld-std --port 5000 --config "configs/tasks/alfworld.yaml" > std.log 2>&1 &

    python -m src.server.task_server_adapter webshop-dev --port 5000 --config "configs/tasks/webshop.yaml" > std.log 2>&1 &
"""

import argparse
import asyncio
import os
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid
import time
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel

from src.configs import ConfigLoader
from src.server.task import Task, Session
from src.typings import AgentOutput, AgentOutputStatus, SampleStatus


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


def _resolve_container_image(task_name: str, container_runtime: str, image: str) -> str:
    """Resolve runtime-specific image references."""
    if container_runtime == "podman" and task_name.startswith("alfworld"):
        if image in {
            "longinyu/agentbench-alfworld",
            "longinyu/agentbench-alfworld:latest",
            "docker.io/longinyu/agentbench-alfworld",
        }:
            return "docker.io/longinyu/agentbench-alfworld:latest"
    if container_runtime == "podman" and task_name.startswith("webshop"):
        if image in {
            "longinyu/agentbench-webshop",
            "longinyu/agentbench-webshop:latest",
            "docker.io/longinyu/agentbench-webshop",
        }:
            return "docker.io/longinyu/agentbench-webshop:latest"
    return image


# ============================================================================
# Request/Response Models
# ============================================================================

class StartEpisodeRequest(BaseModel):
    sample_id: Union[str, int]  # Support both string (OS task) and int (AlfWorld) indices
    config: Optional[Dict[str, Any]] = {}


class StepEpisodeRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]


class CancelEpisodeRequest(BaseModel):
    episode_id: str


class Observation(BaseModel):
    type: str
    content: Union[str, List[Dict[str, str]]]  # Can be string or list of message dicts for full history

class EpisodeResponse(BaseModel):
    episode_id: str
    observation: Optional[Observation]
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


# ============================================================================
# Episode State Management
# ============================================================================

class EpisodeState:
    """Tracks state of a running episode."""

    def __init__(
        self,
        episode_id: str,
        sample_id: str,
        session: Session,
        task_handle: asyncio.Task,
        max_turns: int
    ):
        self.episode_id = episode_id
        self.sample_id = sample_id
        self.session = session
        self.task_handle = task_handle
        self.max_turns = max_turns
        self.turn = 0
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.done = False
        self.final_result = None

    def update_activity(self):
        self.last_activity = datetime.now()

    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if episode has been inactive too long."""
        return (datetime.now() - self.last_activity).total_seconds() > timeout_seconds


# ============================================================================
# Task Server Adapter
# ============================================================================

class TaskServerAdapter:
    """Adapter that wraps an AgentBench Task with the standard API."""

    def __init__(self, task: Task):
        self.task = task
        self.episodes: Dict[str, EpisodeState] = {}
        self.cleanup_interval = 60  # Clean up expired episodes every 60s
        self._cleanup_task = None
        self.initial_observation_timeout = float(
            os.environ.get("AGENTBENCH_INITIAL_OBS_TIMEOUT", "180")
        )

    async def initialize(self):
        """Start background tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self):
        """Cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all running episodes
        for ep_id in list(self.episodes.keys()):
            await self.cancel_episode(ep_id)

        # Release task resources
        self.task.release()

    async def _cleanup_loop(self):
        """Background task to clean up expired episodes."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_episodes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup loop: {e}")

    async def _cleanup_expired_episodes(self):
        """Remove episodes that have been inactive."""
        now = datetime.now()
        expired = [
            ep_id for ep_id, ep in self.episodes.items()
            if ep.is_expired()
        ]

        for ep_id in expired:
            print(f"Cleaning up expired episode: {ep_id}")
            await self.cancel_episode(ep_id)

    def get_task_info(self) -> Dict[str, Any]:
        """Get task metadata."""
        indices = self.task.get_indices()
        return {
            "name": self.task.name,
            "num_samples": len(indices),
            "max_episode_length": getattr(self.task, 'round_limit', 20),
            "observation_type": "text",
            "action_type": "text",
            "description": f"AgentBench {self.task.name} task"
        }

    async def start_episode(
        self,
        sample_id: Union[str, int],
        config: Dict[str, Any]
    ) -> EpisodeResponse:
        """Start a new episode."""
        # Validate sample exists
        indices = self.task.get_indices()
        # print(f"sample_id = {sample_id}, type of sample_id = {type(sample_id)}, indices = {indices}")
        print(f"sample_id = {sample_id}, type of sample_id = {type(sample_id)}, number of indices = {len(indices)}")

        # Normalize sample_id to match the type in indices
        # OS task uses string indices, AlfWorld uses int indices
        if indices and isinstance(indices[0], int) and isinstance(sample_id, str):
            try:
                sample_id = int(sample_id)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid sample_id '{sample_id}': expected integer for this task"
                )
        elif indices and isinstance(indices[0], str) and isinstance(sample_id, int):
            sample_id = str(sample_id)

        if sample_id not in indices:
            raise HTTPException(
                status_code=404,
                detail=f"Sample {sample_id} not found in task {self.task.name}"
            )

        episode_id = str(uuid.uuid4())

        # Create session (AgentBench's abstraction)
        session = Session()

        # Start task execution in background
        async def task_wrapper():
            """Wrapper to capture task result."""
            try:
                result = await self.task.start_sample(sample_id, session)
                return result
            except SystemExit as e:
                # Catch SystemExit to prevent server crash (e.g., from fast_downward PDDL errors)
                error_msg = str(e)
                print(f"SystemExit caught for episode {episode_id}: {error_msg}")
                print(traceback.format_exc())
                # Convert to a regular exception so it doesn't crash the server
                raise RuntimeError(f"Task failed with SystemExit: {error_msg}") from e
            except Exception as e:
                print(f"Task error for episode {episode_id}: {e}")
                print(traceback.format_exc())
                raise

        task_handle = asyncio.create_task(task_wrapper())

        # Wait for initial observation (task sends first prompt)
        # Also monitor task_handle in case the task fails during initialization
        try:
            agent_pull_coro = session.controller.agent_pull()
            agent_pull_task = asyncio.create_task(agent_pull_coro)

            done_tasks, pending_tasks = await asyncio.wait(
                [agent_pull_task, task_handle],
                timeout=self.initial_observation_timeout,
                return_when=asyncio.FIRST_COMPLETED
            )

            if task_handle in done_tasks:
                # Task finished early (error during initialization)
                agent_pull_task.cancel()
                try:
                    await agent_pull_task
                except asyncio.CancelledError:
                    pass

                task_exc = task_handle.exception()
                if task_exc is not None:
                    tb_lines = traceback.format_exception(
                        type(task_exc), task_exc, task_exc.__traceback__
                    )
                    tb_str = "".join(tb_lines).strip() if tb_lines else ""
                    print(
                        f"Task failed during initialization for episode={episode_id}, "
                        f"sample_id={sample_id}, exc_type={type(task_exc).__name__}, exc={task_exc}"
                    )
                    if tb_str:
                        print(tb_str)
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "message": (
                                "Task failed during initialization before first observation"
                            ),
                            "exception_type": type(task_exc).__name__,
                            "exception": str(task_exc),
                            "traceback": tb_str,
                            "sample_id": sample_id,
                            "episode_id": episode_id,
                        },
                    )

                task_result = task_handle.result()
                # Task completed without providing initial observation - unusual
                print(
                    f"Task finished during initialization without first observation: "
                    f"episode={episode_id}, sample_id={sample_id}, "
                    f"status={task_result.status}, result={task_result.result}"
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": (
                            "Task completed during initialization before first observation"
                        ),
                        "status": str(task_result.status),
                        "result": task_result.result,
                        "sample_id": sample_id,
                        "episode_id": episode_id,
                    },
                )

            elif agent_pull_task in done_tasks:
                # Normal case: got initial observation
                initial_output = agent_pull_task.result()

            elif not done_tasks:
                # Timeout
                agent_pull_task.cancel()
                task_handle.cancel()
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Task failed to provide initial observation within "
                        f"{self.initial_observation_timeout:.0f}s"
                    ),
                )

        except asyncio.TimeoutError:
            task_handle.cancel()
            raise HTTPException(
                status_code=500,
                detail=(
                    "Task failed to provide initial observation within "
                    f"{self.initial_observation_timeout:.0f}s"
                ),
            )

        # Create episode state
        max_turns = getattr(self.task, 'round_limit', 20)
        episode = EpisodeState(
            episode_id=episode_id,
            sample_id=sample_id,
            session=session,
            task_handle=task_handle,
            max_turns=max_turns
        )
        self.episodes[episode_id] = episode

        # Send full conversation history as initial observation
        # This includes system prompt, one-shot examples, and task description
        if initial_output.history and len(initial_output.history) > 0:
            # Convert session history to message format
            initial_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in initial_output.history
            ]
        else:
            initial_messages = []

        return EpisodeResponse(
            episode_id=episode_id,
            observation=Observation(type="messages", content=initial_messages),
            reward=0.0,
            done=False,
            info={
                "max_turns": max_turns,
                "sample_id": sample_id,
                "turn": 0
            }
        )

    async def step_episode(
        self,
        episode_id: str,
        action: Dict[str, Any]
    ) -> EpisodeResponse:
        """Execute an action in the episode."""
        if episode_id not in self.episodes:
            raise HTTPException(
                status_code=404,
                detail=f"Episode {episode_id} not found"
            )

        episode = self.episodes[episode_id]
        episode.update_activity()

        if episode.done:
            raise HTTPException(
                status_code=400,
                detail=f"Episode {episode_id} is already done"
            )

        # Extract action content
        action_content = action.get("content", "")
        # print(action_content)

        # Send action to task via AgentBench's session controller
        agent_output = AgentOutput(
            status=AgentOutputStatus.NORMAL,
            content=action_content
        )

        try:
            # Create the agent_pull coroutine
            agent_pull_coro = episode.session.controller.agent_pull(agent_output)
            agent_pull_task = asyncio.create_task(agent_pull_coro)

            # Wait for either agent_pull to complete OR the task to finish early
            # This handles cases where the task returns early (e.g., repeat action check)
            done_tasks, pending_tasks = await asyncio.wait(
                [agent_pull_task, episode.task_handle],
                timeout=180.0,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Check what completed
            if episode.task_handle in done_tasks:
                # Task finished early (e.g., repeat action check, max steps, etc.)
                # Cancel the agent_pull wait
                agent_pull_task.cancel()
                try:
                    await agent_pull_task
                except asyncio.CancelledError:
                    pass

                # Get the task result
                try:
                    task_result = episode.task_handle.result()
                    print(f"Task finished early with status: {task_result.status}")

                    # Mark episode as done and return final result
                    episode.done = True

                    # Extract reward from result
                    # Support both "result" key (AlfWorld) and "reward" key (WebShop)
                    reward = 0.0
                    info = {"turn": episode.turn + 1, "status": str(task_result.status)}
                    if task_result.result:
                        if isinstance(task_result.result, dict):
                            # Try "result" first (AlfWorld), then "reward" (WebShop)
                            reward_value = task_result.result.get("result")
                            if reward_value is None:
                                reward_value = task_result.result.get("reward", 0.0)
                            reward = float(reward_value)

                            # For success: use reward > 0 as indicator if not explicitly provided
                            success = task_result.result.get("success")
                            if success is None:
                                success = reward > 0.0
                            info["success"] = success

                    # Cleanup
                    del self.episodes[episode_id]

                    return EpisodeResponse(
                        episode_id=episode_id,
                        observation=None,
                        reward=reward,
                        done=True,
                        info=info
                    )
                except Exception as e:
                    print(f"Task error: {e}")
                    del self.episodes[episode_id]
                    raise HTTPException(
                        status_code=500,
                        detail=f"Task failed: {e}"
                    )

            elif agent_pull_task in done_tasks:
                # Normal case: agent_pull completed
                task_output = agent_pull_task.result()

            elif not done_tasks:
                # Timeout - nothing completed
                agent_pull_task.cancel()
                raise HTTPException(
                    status_code=500,
                    detail="Task did not respond within 90s"
                )

        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=500,
                detail="Task did not respond within 90s"
            )

        episode.turn += 1

        # Check if episode is done
        done = task_output.status != SampleStatus.RUNNING
        episode.done = done

        # Build response
        observation = None
        reward = 0.0
        info = {"turn": episode.turn}

        if done:
            # Episode finished - get final result
            try:
                final_result = await asyncio.wait_for(
                    episode.task_handle,
                    timeout=5.0
                )
                episode.final_result = final_result

                # Extract reward from result
                # Support both "result" key (AlfWorld) and "reward" key (WebShop)
                if hasattr(final_result, 'result') and final_result.result:
                    if isinstance(final_result.result, dict):
                        # Try "result" first (AlfWorld convention), then "reward" (WebShop convention)
                        reward_value = final_result.result.get("result")
                        if reward_value is None:
                            reward_value = final_result.result.get("reward", 0.0)
                        reward = float(reward_value)

                        # For success: use reward > 0 as indicator if not explicitly provided
                        success = final_result.result.get("success")
                        if success is None:
                            success = reward > 0.0
                        info["success"] = success
                        info["answer"] = final_result.result.get("answer", "")

                info["status"] = str(task_output.status)
                info["num_turns"] = episode.turn

            except asyncio.TimeoutError:
                print(f"Warning: Task did not complete within timeout for {episode_id}")
                info["status"] = "timeout"

            # Cleanup
            del self.episodes[episode_id]

        else:
            # Extract next observation from history
            if task_output.history and len(task_output.history) > 0:
                obs_content = task_output.history[-1].content
                observation = Observation(type="text", content=obs_content)

        return EpisodeResponse(
            episode_id=episode_id,
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )

    async def cancel_episode(self, episode_id: str) -> Dict[str, str]:
        """Cancel a running episode."""
        if episode_id not in self.episodes:
            raise HTTPException(
                status_code=404,
                detail=f"Episode {episode_id} not found"
            )

        episode = self.episodes[episode_id]

        # Send cancel signal to task
        try:
            episode.session.controller.env_input = AgentOutput(
                status=AgentOutputStatus.CANCELLED
            )
            episode.session.controller.env_signal.release()
        except Exception as e:
            print(f"Error sending cancel signal: {e}")

        # Cancel task
        episode.task_handle.cancel()
        try:
            await asyncio.wait_for(episode.task_handle, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        # Remove from tracking
        del self.episodes[episode_id]

        return {"status": "cancelled", "episode_id": episode_id}


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app(task: Task) -> FastAPI:
    """Create FastAPI application with task server adapter."""
    app = FastAPI(
        title="AgentBench Task Server",
        description="Standard task server API for AgentBench tasks",
        version="1.0.0"
    )

    adapter = TaskServerAdapter(task)

    @app.on_event("startup")
    async def startup():
        await adapter.initialize()

    @app.on_event("shutdown")
    async def shutdown():
        await adapter.shutdown()

    # Create router with /api prefix
    router = APIRouter(prefix="/api")

    @router.get("/task/info")
    async def get_task_info():
        """Get task metadata."""
        return adapter.get_task_info()

    @router.post("/episode/start", response_model=EpisodeResponse)
    async def start_episode(request: StartEpisodeRequest):
        """Start a new episode."""
        return await adapter.start_episode(request.sample_id, request.config)

    @router.post("/episode/step", response_model=EpisodeResponse)
    async def step_episode(request: StepEpisodeRequest):
        """Execute an action in the episode."""
        return await adapter.step_episode(request.episode_id, request.action)

    @router.post("/episode/cancel")
    async def cancel_episode(request: CancelEpisodeRequest):
        """Cancel a running episode."""
        return await adapter.cancel_episode(request.episode_id)

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "active_episodes": len(adapter.episodes),
            "task_name": adapter.task.name
        }

    app.include_router(router)

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run AgentBench task as a standard task server"
    )
    parser.add_argument(
        "task_name",
        type=str,
        help="Task name from config (e.g., 'os-dev', 'os-std')"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/tasks/os.yaml",
        help="Path to task config file"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port to run server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--container-runtime",
        type=str,
        choices=["docker", "podman"],
        default="docker",
        help="Container runtime to use when task config has a container image"
    )
    parser.add_argument(
        "--skip-container",
        "--skip-docker",
        dest="skip_container",
        action="store_true",
        help="Skip container-runtime detection (used when already running inside a container)"
    )

    args = parser.parse_args()

    # Load task configuration
    print(f"Loading task configuration from {args.config}...")
    config_loader = ConfigLoader()
    config = config_loader.load_from(args.config)

    if args.task_name not in config:
        available = ", ".join(config.keys())
        raise ValueError(
            f"Task '{args.task_name}' not found in config. "
            f"Available tasks: {available}"
        )

    task_config = config[args.task_name]

    # Check if task requires a container runtime (skip if already inside container)
    if not args.skip_container and "docker" in task_config and "image" in task_config["docker"]:
        docker_config = task_config["docker"]
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
        container_runtime = args.container_runtime

        print(f"\n{'='*70}")
        print(f"Task '{args.task_name}' requires container runtime")
        print(f"{'='*70}")
        print(f"Container runtime: {container_runtime}")
        container_image = _resolve_container_image(
            task_name=args.task_name,
            container_runtime=container_runtime,
            image=docker_config["image"],
        )
        print(f"Container image: {container_image}")
        print(f"Starting container...")
        print(f"{'='*70}\n")

        # Launch container that runs the task server adapter inside
        # Mount to /root/workspace (not /AgentBench) to preserve Docker's built-in data
        container_name = f"agentbench-adapter-{args.port}"
        host_tmp_root = os.environ.get("AGENTBENCH_HOST_TMP_ROOT", "/tmp/agentbench-adapter-tmp")
        adapter_tmp_dir = os.path.join(host_tmp_root, container_name)
        # Give each adapter an isolated host-backed /tmp on local storage.
        # Avoid repo-mounted NFS paths here because tempfile cleanup in
        # fast_downward can fail with Errno 39 on network filesystems.
        shutil.rmtree(adapter_tmp_dir, ignore_errors=True)
        os.makedirs(adapter_tmp_dir, mode=0o1777, exist_ok=True)
        os.chmod(adapter_tmp_dir, 0o1777)

        docker_cmd = _build_docker_base_cmd(container_runtime) + [
            "run",
            "--rm",
            "--name", container_name,
            "--network", "host",
            "--security-opt", "label=disable",
            "--add-host", "host.docker.internal:host-gateway",
            "-v", f"{project_root}:/root/workspace",
            "-v", f"{adapter_tmp_dir}:/tmp",
            "-e", "TMPDIR=/tmp",
            "-e", "TMP=/tmp",
            "-e", "TEMP=/tmp",
            "-w", "/root/workspace",
            container_image,
            "bash",
            "-c",
            (docker_config.get("command", "") +
             f" python -m src.server.task_server_adapter {args.task_name} "
             f"--config {args.config} "
             f"--port {args.port} "
             f"--host {args.host} "
             f"--skip-docker")
        ]
        # docker_cmd = [
        #     "docker",
        #     "run",
        #     "--rm",
        #     "--name", container_name,
        #     "-p", f"{args.port}:{args.port}",
        #     "--add-host", "host.docker.internal:host-gateway",
        #     "-v", f"{project_root}:/root/workspace",
        #     "-w", "/root/workspace",
        #     docker_config["image"],
        #     "bash",
        #     "-c",
        #     (docker_config.get("command", "") +
        #      f" python -m src.server.task_server_adapter {args.task_name} "
        #      f"--config {args.config} "
        #      f"--port {args.port} "
        #      f"--host {args.host} "
        #      f"--skip-docker")
        # ]

        print(f"Running: {shlex.join(docker_cmd)}\n")

        # Execute Docker command and stream output
        try:
            proc = subprocess.Popen(docker_cmd)
            proc.wait()
        except FileNotFoundError:
            print(
                f"Error: container runtime '{container_runtime}' not found in PATH. "
                "Install it or use --container-runtime docker."
            )
            raise SystemExit(1)
        except KeyboardInterrupt:
            print("\nShutting down container...")
            proc.terminate()
            proc.wait()

        return

    # Create task instance (for non-containerized tasks)
    from src.typings import InstanceFactory
    print(f"Initializing task '{args.task_name}'...")
    task = InstanceFactory.parse_obj(task_config).create()

    # Create and run app
    app = create_app(task)

    print(f"\n{'='*70}")
    print(f"Starting Task Server: {args.task_name}")
    print(f"{'='*70}")
    print(f"Server URL: http://{args.host}:{args.port}")
    print(f"API Base:   http://{args.host}:{args.port}/api")
    print(f"Health:     http://{args.host}:{args.port}/api/health")
    print(f"Task Info:  http://{args.host}:{args.port}/api/task/info")
    print(f"{'='*70}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
