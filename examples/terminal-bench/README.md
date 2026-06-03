# Terminal-Bench Recipes

Harbor-backed Terminal-Bench eval and RL recipes.

Run the Terminal-Bench 2 eval recipe from the repo root:

```bash
bash examples/terminal-bench/terminal-bench-2-qwen3-8b/scripts/run_terminal-bench-2-qwen3-8b.sh
```

Run the Harbor RL Podman recipe from the repo root:

```bash
bash examples/terminal-bench/terminal-bench-rl-qwen3-14b-podman-test/scripts/run_terminal-bench-rl-qwen3-14b-podman-test.sh
```

The Podman Harbor environment helper is `examples/terminal-bench/harbor_podman_env.py`.

Complete guidance: [`docs/en/recipes/code.md`](../../docs/en/recipes/code.md#terminal-bench-2--harbor-setup).

---
**GPU Resources**

The Terminal-Bench 2 eval recipe defaults to one 8xH100 node. The Harbor RL
Podman test recipe defaults to 2 GPUs for inference and 2 GPUs for training.
