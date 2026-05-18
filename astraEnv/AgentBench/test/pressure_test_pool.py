"""
Pressure test for the Adapter Pool Controller.

Fires 128 concurrent episodes (all sample_id=0) through the controller,
alternating actions each step, and lets them run until done or 50 steps.
Designed to stress the 4-adapter pool with restart-after-episodes=8.

Usage:
    python test/pressure_test_pool.py
    python test/pressure_test_pool.py --host localhost --port 5000 --episodes 128 --steps 15
"""

import argparse
import asyncio
import time
from collections import Counter

import aiohttp


async def run_episode(
    session: aiohttp.ClientSession,
    base_url: str,
    sample_id: int,
    max_steps: int,
    episode_idx: int,
):
    """Run a single episode: start -> alternate actions -> until done or max_steps."""
    tag = f"[Ep {episode_idx:3d}]"
    episode_id = None

    try:
        # --- Start episode ---
        t0 = time.monotonic()
        async with session.post(
            f"{base_url}/api/episode/start",
            json={"sample_id": sample_id, "config": {}},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"{tag} START FAILED ({resp.status}): {body}")
                return {"episode_idx": episode_idx, "status": "start_failed", "error": body}
            data = await resp.json()

        episode_id = data["episode_id"]
        start_dur = time.monotonic() - t0
        print(f"{tag} started  id={episode_id[:8]}...  ({start_dur:.2f}s)")

        # --- Step loop ---
        for step in range(1, max_steps + 1):
            if data.get("done"):
                reward = data.get("reward", 0.0)
                print(f"{tag} done at step {step - 1}, reward={reward}")
                return {
                    "episode_idx": episode_idx,
                    "status": "done",
                    "steps": step - 1,
                    "reward": reward,
                }

            # Alternate actions
            if step % 2 == 1:
                action_content = "ACTION: go to desk 1"
            else:
                action_content = "ACTION: go to bed 1"

            t1 = time.monotonic()
            async with session.post(
                f"{base_url}/api/episode/step",
                json={
                    "episode_id": episode_id,
                    "action": {"content": action_content},
                },
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"{tag} STEP {step} FAILED ({resp.status}): {body}")
                    return {
                        "episode_idx": episode_idx,
                        "status": "step_failed",
                        "steps": step - 1,
                        "error": body,
                    }
                data = await resp.json()

            step_dur = time.monotonic() - t1
            obs_preview = ""
            if data.get("observation") and data["observation"].get("content"):
                content = data["observation"]["content"]
                if isinstance(content, str):
                    obs_preview = content[:60].replace("\n", " ")
            print(f"{tag} step {step:2d}  ({step_dur:.2f}s)  done={data.get('done')}  obs={obs_preview!r}")

            if data.get("done"):
                reward = data.get("reward", 0.0)
                print(f"{tag} done at step {step}, reward={reward}")
                return {
                    "episode_idx": episode_idx,
                    "status": "done",
                    "steps": step,
                    "reward": reward,
                }

        # Reached max steps without finishing — cancel
        print(f"{tag} max steps reached, cancelling")
        async with session.post(
            f"{base_url}/api/episode/cancel",
            json={"episode_id": episode_id},
        ) as resp:
            pass

        return {
            "episode_idx": episode_idx,
            "status": "max_steps",
            "steps": max_steps,
        }

    except Exception as e:
        print(f"{tag} EXCEPTION: {e}")
        if episode_id:
            try:
                await session.post(
                    f"{base_url}/api/episode/cancel",
                    json={"episode_id": episode_id},
                )
            except Exception:
                pass
        return {"episode_idx": episode_idx, "status": "exception", "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Pressure test for adapter pool controller")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=128, help="Total number of episodes to launch")
    parser.add_argument("--steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Max concurrent episodes (0 = all at once)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    num_episodes = args.episodes
    max_steps = args.steps

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # --- Health check ---
        print(f"Checking health at {base_url}/api/health ...")
        async with session.get(f"{base_url}/api/health") as resp:
            health = await resp.json()
            print(f"Health: {health}\n")

        # --- Task info ---
        async with session.get(f"{base_url}/api/task/info") as resp:
            info = await resp.json()
            print(f"Task info: {info}\n")

        print(f"Launching {num_episodes} episodes (sample_id=0, max_steps={max_steps})")
        print(f"Adapter pool: {health.get('num_adapters', '?')} adapters")
        print("=" * 70)

        t_start = time.monotonic()

        if args.concurrency > 0:
            # Throttled: use a semaphore to limit concurrency
            sem = asyncio.Semaphore(args.concurrency)

            async def throttled(idx):
                async with sem:
                    return await run_episode(session, base_url, 0, max_steps, idx)

            tasks = [throttled(i) for i in range(num_episodes)]
        else:
            # All at once
            tasks = [
                run_episode(session, base_url, 0, max_steps, i)
                for i in range(num_episodes)
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        t_total = time.monotonic() - t_start

        # --- Final health ---
        print("\n" + "=" * 70)
        print("FINAL POOL STATUS")
        print("=" * 70)
        async with session.get(f"{base_url}/api/health") as resp:
            final_health = await resp.json()
            for port, adapter_info in final_health.get("adapters", {}).items():
                print(f"  Adapter {port}: status={adapter_info['status']}  "
                      f"total_episodes={adapter_info['total_episodes']}  "
                      f"restarts={adapter_info['restart_count']}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    status_counts = Counter()
    total_steps = 0
    total_reward = 0.0

    for r in results:
        if isinstance(r, Exception):
            print(f"  Exception: {r}")
            status_counts["exception"] += 1
        elif isinstance(r, dict):
            status = r.get("status", "unknown")
            status_counts[status] += 1
            total_steps += r.get("steps", 0)
            total_reward += r.get("reward", 0.0)
            if status in ("start_failed", "step_failed", "exception"):
                print(f"  Episode {r['episode_idx']}: {status} - {r.get('error', '')[:100]}")

    print(f"\nTotal episodes:    {num_episodes}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:15s}:  {count}")
    print(f"\nTotal steps:       {total_steps}")
    print(f"Total reward:      {total_reward:.1f}")
    print(f"Avg steps/episode: {total_steps / num_episodes:.1f}")
    print(f"\nTotal time:        {t_total:.2f}s")
    print(f"Avg per episode:   {t_total / num_episodes:.2f}s")
    print(f"Throughput:        {num_episodes / t_total:.2f} episodes/s")


if __name__ == "__main__":
    asyncio.run(main())