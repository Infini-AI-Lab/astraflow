"""
Pressure test for ALFWorld task server adapter.

Starts 16 episodes concurrently, interacts with each for 15 steps,
then cancels all episodes. All interactions are async.

Usage:
    python test/pressure_test_adapter.py
    python test/pressure_test_adapter.py --host localhost --port 5000 --episodes 16 --steps 15
"""

import argparse
import asyncio
import time

import aiohttp


async def run_episode(
    session: aiohttp.ClientSession,
    base_url: str,
    sample_id: int,
    num_steps: int,
    episode_idx: int,
):
    """Run a single episode: start -> step N times -> cancel."""
    tag = f"[Episode {episode_idx} sample={sample_id}]"
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
                return {"episode_idx": episode_idx, "error": f"start failed: {resp.status}"}
            data = await resp.json()

        episode_id = data["episode_id"]
        start_dur = time.monotonic() - t0
        print(f"{tag} started  id={episode_id}...  ({start_dur:.2f}s)")

        # --- Step loop ---
        for step in range(1, num_steps + 1):
            if data.get("done"):
                print(f"{tag} done early at step {step - 1}")
                return {
                    "episode_idx": episode_idx,
                    "steps_completed": step - 1,
                    "done_early": True,
                    "reward": data.get("reward", 0.0),
                }

            t1 = time.monotonic()
            if step % 2 == 1:
                json={
                    "episode_id": episode_id,
                    "action": {"content": "ACTION: go to desk 1"},
                }
            else:
                json={
                    "episode_id": episode_id,
                    "action": {"content": "ACTION: go to bed 1"},
                }
            async with session.post(
                f"{base_url}/api/episode/step",
                json=json,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"{tag} STEP {step} FAILED ({resp.status}): {body}")
                    return {
                        "episode_idx": episode_idx,
                        "steps_completed": step - 1,
                        "error": f"step {step} failed: {resp.status}",
                    }
                data = await resp.json()

            step_dur = time.monotonic() - t1
            obs_preview = ""
            if data.get("observation") and data["observation"].get("content"):
                content = data["observation"]["content"]
                if isinstance(content, str):
                    obs_preview = content[:80].replace("\n", " ")
            print(f"{tag} step {step:2d}  ({step_dur:.2f}s)  done={data.get('done')}  obs={obs_preview!r}")

            if data.get("done"):
                print(f"{tag} done at step {step}, reward={data.get('reward', 0.0)}")
                return {
                    "episode_idx": episode_idx,
                    "steps_completed": step,
                    "done_early": True,
                    "reward": data.get("reward", 0.0),
                }

        # --- Cancel episode ---
        t2 = time.monotonic()
        async with session.post(
            f"{base_url}/api/episode/cancel",
            json={"episode_id": episode_id},
        ) as resp:
            cancel_status = resp.status
            cancel_body = await resp.text()

        cancel_dur = time.monotonic() - t2
        print(f"{tag} cancelled ({cancel_dur:.2f}s) status={cancel_status}")

        return {
            "episode_idx": episode_idx,
            "steps_completed": num_steps,
            "cancelled": True,
        }

    except Exception as e:
        print(f"{tag} EXCEPTION: {e}")
        # Try to cancel if we have an episode_id
        if episode_id:
            try:
                await session.post(
                    f"{base_url}/api/episode/cancel",
                    json={"episode_id": episode_id},
                )
            except Exception:
                pass
        return {"episode_idx": episode_idx, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Pressure test for ALFWorld adapter")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=16, help="Number of concurrent episodes")
    parser.add_argument("--steps", type=int, default=15, help="Steps per episode before cancel")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    num_episodes = args.episodes
    num_steps = args.steps

    # Health check first
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        print(f"Checking health at {base_url}/api/health ...")
        async with session.get(f"{base_url}/api/health") as resp:
            health = await resp.json()
            print(f"Health: {health}\n")

        # Get task info
        async with session.get(f"{base_url}/api/task/info") as resp:
            info = await resp.json()
            num_samples = info.get("num_samples", 0)
            print(f"Task info: {info}")
            print(f"Available samples: {num_samples}\n")

        if num_samples == 0:
            print("ERROR: No samples available")
            return

        # Assign sample IDs (wrap around if more episodes than samples)
        sample_ids = [i % num_samples for i in range(num_episodes)]

        print(f"Starting {num_episodes} episodes with {num_steps} steps each...")
        print(f"Sample IDs: {sample_ids}")
        print("=" * 70)

        t_start = time.monotonic()

        # Launch all episodes concurrently
        tasks = [
            run_episode(session, base_url, 4649+sample_id, num_steps, idx)
            for idx, sample_id in enumerate(sample_ids)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        t_total = time.monotonic() - t_start

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    errors = 0
    done_early = 0
    cancelled = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"  Exception: {r}")
            errors += 1
        elif isinstance(r, dict):
            if r.get("error"):
                print(f"  Episode {r['episode_idx']}: ERROR - {r['error']}")
                errors += 1
            elif r.get("done_early"):
                print(f"  Episode {r['episode_idx']}: done early at step {r['steps_completed']}, reward={r.get('reward', 0)}")
                done_early += 1
            elif r.get("cancelled"):
                print(f"  Episode {r['episode_idx']}: completed {r['steps_completed']} steps, cancelled")
                cancelled += 1

    print(f"\nTotal episodes:  {num_episodes}")
    print(f"  Cancelled:     {cancelled}")
    print(f"  Done early:    {done_early}")
    print(f"  Errors:        {errors}")
    print(f"\nTotal time:      {t_total:.2f}s")
    print(f"Avg per episode: {t_total / num_episodes:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())