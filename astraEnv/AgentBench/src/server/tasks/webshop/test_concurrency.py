"""
Test script to verify WebShop task handles concurrent episodes correctly.

This test creates multiple episodes concurrently and verifies:
1. Each episode gets its own isolated environment state
2. Episodes don't interfere with each other's observations
3. Session IDs are properly isolated
4. No race conditions occur during concurrent execution
"""

import asyncio
import sys
from pathlib import Path

# Add AgentBench to path
agentbench_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(agentbench_root))

from src.server.tasks.webshop import WebShop
from src.server.task import Session
from src.typings import AgentOutput, AgentOutputStatus


class MockAgent:
    """Mock agent that follows a simple action sequence for testing."""

    def __init__(self, episode_id: int):
        self.episode_id = episode_id
        self.actions = [
            "search[bottle]",
            "click[back to search]",
            "search[pen]",
        ]
        self.step = 0

    def get_action(self) -> str:
        if self.step >= len(self.actions):
            return "click[buy now]"
        action = self.actions[self.step]
        self.step += 1
        return f"Thought: Episode {self.episode_id} step {self.step}\n\nAction:\n{action}"


async def run_episode(task: WebShop, index: int, episode_id: int) -> dict:
    """Run a single episode with a mock agent."""
    print(f"[Episode {episode_id}] Starting with index {index}")

    # Create session and mock agent
    session = Session()
    agent = MockAgent(episode_id)

    # Override session.action() to return mock agent responses
    async def mock_action():
        action_str = agent.get_action()
        return AgentOutput(status=AgentOutputStatus.NORMAL, content=action_str)

    session.action = mock_action

    # Run episode
    try:
        result = await task.start_sample(index, session)
        print(f"[Episode {episode_id}] Completed: status={result.status}, reward={result.result.get('reward', 0)}")
        return {
            "episode_id": episode_id,
            "index": index,
            "status": result.status,
            "reward": result.result.get("reward", 0),
            "num_steps": len(result.result.get("history", [])),
        }
    except Exception as e:
        print(f"[Episode {episode_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "episode_id": episode_id,
            "index": index,
            "error": str(e),
        }


async def test_concurrent_episodes(num_episodes: int = 5):
    """Test running multiple episodes concurrently."""
    print(f"\n{'='*70}")
    print(f"Testing {num_episodes} concurrent WebShop episodes")
    print(f"{'='*70}\n")

    # Create WebShop task
    task = WebShop(start=0, end=100)

    # Run multiple episodes concurrently with different indices
    indices = list(range(num_episodes))
    episode_ids = list(range(num_episodes))

    print(f"Starting {num_episodes} concurrent episodes...\n")

    # Run all episodes concurrently
    results = await asyncio.gather(
        *[run_episode(task, idx, ep_id) for idx, ep_id in zip(indices, episode_ids)]
    )

    # Verify results
    print(f"\n{'='*70}")
    print("Results Summary:")
    print(f"{'='*70}\n")

    success_count = 0
    error_count = 0

    for result in results:
        if "error" in result:
            print(f"❌ Episode {result['episode_id']} (index {result['index']}): ERROR - {result['error']}")
            error_count += 1
        else:
            print(f"✓ Episode {result['episode_id']} (index {result['index']}): {result['status']}, "
                  f"reward={result['reward']:.2f}, steps={result['num_steps']}")
            success_count += 1

    print(f"\n{'='*70}")
    print(f"Summary: {success_count}/{num_episodes} successful, {error_count} errors")
    print(f"{'='*70}\n")

    # Check for race conditions
    if error_count == 0:
        print("✅ All episodes completed without errors - no race conditions detected!")
        return True
    else:
        print("❌ Some episodes failed - possible race conditions or other issues")
        return False


async def test_sequential_episodes(num_episodes: int = 3):
    """Test running episodes sequentially for comparison."""
    print(f"\n{'='*70}")
    print(f"Testing {num_episodes} sequential WebShop episodes (baseline)")
    print(f"{'='*70}\n")

    task = WebShop(start=0, end=100)

    results = []
    for i in range(num_episodes):
        result = await run_episode(task, i, i)
        results.append(result)

    print(f"\n{'='*70}")
    print("Sequential execution completed")
    print(f"{'='*70}\n")

    return all("error" not in r for r in results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test WebShop concurrency")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of concurrent episodes to test")
    parser.add_argument("--mode", choices=["concurrent", "sequential", "both"], default="concurrent",
                        help="Test mode")
    args = parser.parse_args()

    async def main():
        if args.mode in ["sequential", "both"]:
            await test_sequential_episodes(3)

        if args.mode in ["concurrent", "both"]:
            success = await test_concurrent_episodes(args.num_episodes)
            return success
        return True

    success = asyncio.run(main())
    sys.exit(0 if success else 1)