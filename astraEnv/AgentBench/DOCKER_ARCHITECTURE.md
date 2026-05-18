# Docker Architecture in Task Server Adapter

## Overview

The task_server_adapter now supports **automatic Docker launching** for tasks that require it, matching the behavior of the standard AgentBench framework.

## How It Works

### Detection

When you run:
```bash
python -m src.server.task_server_adapter alfworld-std --port 5000
```

The adapter checks the task configuration for a `docker` section:

```yaml
# In configs/tasks/alfworld.yaml
alfworld-std:
  module: src.server.tasks.alfworld.ALFWorld
  docker:
    image: longinyu/agentbench-alfworld  # <-- Docker config detected!
    command: umask 0; [ -f /root/.setup.sh ] && bash /root/.setup.sh;
  parameters:
    # ... task parameters
```

### Automatic Docker Launch

If `docker.image` is present, the adapter:
1. Launches a Docker container with the specified image
2. Mounts the AgentBench directory into the container
3. Runs the task_server_adapter **inside** the container
4. Forwards the port so you can connect from outside

The command executed:
```bash
docker run --rm \
  -p 5000:5000 \
  --add-host host.docker.internal:host-gateway \
  -v /path/to/AgentBench:/AgentBench \
  -w /AgentBench \
  longinyu/agentbench-alfworld \
  bash -c "umask 0; [...setup...]; python -m src.server.task_server_adapter alfworld-std --port 5000"
```

### Direct Execution

If no `docker` section exists, the adapter runs the task directly in your current Python environment.

## Comparison: AlfWorld vs OS Interaction

### AlfWorld Architecture

```
Host Machine:
  ├─ You run: python -m src.server.task_server_adapter alfworld-std --port 5000
  │
  └─ Adapter detects Docker requirement
     │
     └─ Launches Docker container (longinyu/agentbench-alfworld)
        │
        └─ Inside container:
           ├─ AlfWorld environment (textworld, alfworld packages)
           ├─ AlfWorld data (/AgentBench/data/alfworld)
           └─ Task server runs on port 5000
              └─ Serves episodes via HTTP API
```

### OS Interaction Architecture

```
Host Machine:
  ├─ You run: python -m src.server.task_server_adapter os-std --port 5000
  │
  └─ Adapter runs directly (no Docker wrapper)
     │
     └─ Task server (on host)
        │
        └─ For each episode:
           ├─ Creates a new Docker container
           ├─ Executes bash commands inside
           └─ Destroys container when done
```

## Key Differences

| Aspect | AlfWorld | OS Interaction |
|--------|----------|----------------|
| **Task Server Location** | Inside Docker container | On host machine |
| **Docker Usage** | One persistent container for all episodes | New container per episode |
| **Why Docker?** | Dependencies (textworld, alfworld, data) | Isolated execution environment for bash |
| **Implementation** | task_server_adapter launches Docker | task.py creates Docker containers |

## Benefits

1. **Consistency**: Same Docker usage as standard AgentBench (`start_task.py`)
2. **Zero Setup**: No need to install textworld/alfworld on host
3. **Data Included**: Docker image contains all AlfWorld data
4. **Isolation**: Task runs in isolated environment

## Usage

Simply run the task_server_adapter - Docker is handled automatically:

```bash
# AlfWorld (runs in Docker automatically)
python -m src.server.task_server_adapter alfworld-std --port 5000

# OS Interaction (runs directly, creates containers for tasks)
python -m src.server.task_server_adapter os-std --port 5000
```

You'll see output like:
```
Loading task configuration from configs/tasks/alfworld.yaml...
======================================================================
Task 'alfworld-std' requires Docker
======================================================================
Docker image: longinyu/agentbench-alfworld
Starting Docker container...
======================================================================

Running: docker run --rm -p 5000:5000 ...
```

## Troubleshooting

### "docker: command not found"

Install Docker on your system.

### "docker: Cannot connect to the Docker daemon"

Start the Docker daemon:
```bash
sudo systemctl start docker
```

### "Unable to find image 'longinyu/agentbench-alfworld:latest'"

Pull the image first:
```bash
docker pull longinyu/agentbench-alfworld
```

### Port already in use

Stop any existing container on that port:
```bash
docker ps  # Find container ID
docker stop <container-id>
```