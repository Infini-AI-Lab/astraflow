from os.path import dirname, realpath

import asyncio
import concurrent.futures
import re
import sys
import threading
import uuid
from typing import Dict, List, Any

sys.path.append(dirname(realpath(__file__)))


from src.server.task import Task, Session
from src.typings import AgentOutputStatus, SampleStatus, TaskOutput
from .web_agent_site.envs.web_agent_text_env import WebAgentTextEnv, SimServer
from .web_agent_site.utils import DEFAULT_FILE_PATH

# Global lock to protect thread-unsafe operations in SimServer
# The Lucene search engine and timing stats are not thread-safe
_webshop_lock = threading.Lock()
# Shared executor for blocking env operations
_webshop_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=64, thread_name_prefix="webshop_env"
)

prompt: str = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, \
you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought:
I think ...

Action:
click[something]
"""


class WebShop(Task):
    """WebShop task with proper concurrency support for parallel episode execution.

    Concurrency Strategy:
    ---------------------
    This implementation supports running multiple episodes concurrently (e.g., with
    AReaL's AsyncTaskRunner) by following these patterns:

    1. **Shared Server**: A single SimServer instance is created in __init__ and shared
       across all episodes. This is safe because:
       - Products, goals, and search engine are read-only after initialization
       - Session state is properly isolated in user_sessions dict (keyed by session_id)

    2. **Per-Episode Environments**: Each start_sample() call creates a new
       WebAgentTextEnv instance with isolated state (session, prev_obs, prev_actions,
       browser, etc.) to prevent race conditions between concurrent episodes.

    3. **Thread-Safe Operations**: The global _webshop_lock protects thread-unsafe
       operations in SimServer (Lucene search engine, timing stats) during env.reset()
       and env.step() calls.

    4. **Unique Session IDs Per Episode**: Session state in SimServer is keyed by
       session_id. We remap each episode to a unique key so concurrent episodes
       using the same sample index do not collide.

    This design matches the pattern used in AlfWorld task and ensures correct behavior
    when multiple episodes run concurrently via asyncio.
    """

    def __init__(self, **configs):
        super().__init__(**configs)
        self.ranging = (configs.pop("start", 0), configs.pop("end", 500))

        # Create a shared SimServer to avoid reloading products/goals for each episode
        # The server properly isolates state by session_id in user_sessions dict
        self.server = SimServer(
            base_url='http://127.0.0.1:3000',
            file_path=DEFAULT_FILE_PATH,
            filter_goals=None,
            limit_goals=-1,
            num_products=None,
            human_goals=True,
            show_attrs=False,
        )
        print(f"WebShop: Initialized shared server with {len(self.server.goals)} goals")

    def get_indices(self) -> List[Any]:
        return list(range(*self.ranging))

    def release(self):
        """Clean up resources when task is done."""
        if hasattr(self, 'server'):
            del self.server

    @staticmethod
    def _build_episode_session_id(index: int) -> str:
        return f"{index}-{uuid.uuid4().hex}"

    def _reset_env_for_episode(self, env: WebAgentTextEnv, index: int, episode_session_id: str):
        """Reset environment with deterministic goal index and unique session key."""
        with _webshop_lock:
            # WebAgentTextEnv.__init__ performs an implicit reset() with a random
            # session ID. Remove it to avoid stale session accumulation.
            bootstrap_session = getattr(env, "session", None)
            if bootstrap_session is not None:
                self.server.user_sessions.pop(str(bootstrap_session), None)

            # Use index for deterministic goal selection.
            env.reset(index)

            # Remap from str(index) to a unique per-episode key so repeated
            # indices can run concurrently without sharing server session state.
            original_session_id = str(index)
            if original_session_id in self.server.user_sessions:
                self.server.user_sessions[episode_session_id] = self.server.user_sessions.pop(
                    original_session_id
                )
            elif getattr(env, "session", None) in self.server.user_sessions:
                fallback_session_id = str(env.session)
                self.server.user_sessions[episode_session_id] = self.server.user_sessions.pop(
                    fallback_session_id
                )
            else:
                raise RuntimeError(
                    f"WebShop reset failed: missing session state for index={index}"
                )

            env.session = episode_session_id
            env.browser.session_id = episode_session_id
            if env.browser.current_url:
                env.browser.current_url = env.browser.current_url.replace(
                    f"/{original_session_id}", f"/{episode_session_id}", 1
                )

            return env.observation

    def _cleanup_episode_session(self, episode_session_id: str):
        """Release per-episode server session state."""
        with _webshop_lock:
            self.server.user_sessions.pop(episode_session_id, None)

    async def start_sample(self, index: int, session: Session) -> TaskOutput:
        history = []
        episode_session_id = self._build_episode_session_id(index)

        # Create a new environment instance per episode to avoid race conditions
        # Pass the shared server which properly isolates sessions
        env = WebAgentTextEnv(
            observation_mode="text_rich",
            human_goals=True,
            server=self.server  # Reuse shared server
        )
        loop = asyncio.get_event_loop()

        try:
            observation = await asyncio.wait_for(
                loop.run_in_executor(
                    _webshop_executor,
                    self._reset_env_for_episode,
                    env,
                    index,
                    episode_session_id,
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            return TaskOutput(
                status=SampleStatus.TASK_ERROR,
                result={"reward": 0.0, "history": history, "error": "env.reset() timeout"},
            )
        except Exception as e:
            return TaskOutput(
                status=SampleStatus.TASK_ERROR,
                result={"reward": 0.0, "history": history, "error": f"env.reset() error: {e}"},
            )

        try:
            session.inject({"role": "user", "content": prompt})
            session.inject({"role": "agent", "content": "Ok."})

            # one shot

            session.inject({'role': 'user', 'content': 'Observation:\n"WebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Search"\n\nAvailable Actions:\n{"has_search_bar": true, "clickables": ["..."]}'})
            session.inject({'role': 'agent', 'content': 'Thought:\nI think I should use the search bar to look for the product I need.\n\nAction:\nsearch[l\'eau d\'issey 6.76 fl oz bottle price < 100.00]'})
            session.inject({'role': 'user', 'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L\'eau d\'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L\'eau D\'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L\'eau D\'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L\'Eau d\'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L\'Eau D\'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L\'eau D\'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L\'eau d\'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L\'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L\'eau d\'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."]}'})
            session.inject({'role': 'agent', 'content': 'Thought:\nI think I should click on the product I need, which is B000VOHH8I.\n\nAction:\nclick[B000VOHH8I]'})
            session.inject({'role': 'user', 'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
            session.inject({'role': 'agent', 'content': 'Thought:\nI think I should click on the \'6.76 fl oz (pack of 1)\' option to select the size I need.\n\nAction:\nclick[6.76 fl oz (pack of 1)]'})
            session.inject({'role': 'user', 'content': 'Observation:\n"Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L\'eau D\'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"\n\nAvailable Actions:\n{"has_search_bar": false, "clickables": ["...", "...", "...", "...", "...", "...", "...", "..."]}'})
            session.inject({'role': 'agent', 'content': 'Thought:\nI think I should click on the \'Buy Now\' button to purchase the product.\n\nAction:\nclick[Buy Now]'})

            reward = 0.0
            finish_reason = SampleStatus.COMPLETED
            for j in range(10):
                available_actions = env.get_available_actions()
                session.inject(
                    {
                        "role": "user",
                        "content": f"Observation:\n{observation}\n\n"
                        f"Available Actions:\n{available_actions}",
                    }
                )
                response = await session.action()
                if response.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                    finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                    break
                response = response.content
                try:
                    action = re.search(
                        r"[Aa]ction: *\n* *((search|click)\[.+?])", response
                    ).group(1)
                except Exception:
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                    action = None
                history.append(
                    {
                        "observation": observation,
                        "available_actions": available_actions,
                        "response": response,
                        "action": action,
                    }
                )
                if not action:
                    reward = 0.0
                    break

                # Run blocking env.step() in shared executor so one long step
                # does not block the adapter event loop.
                def safe_env_step():
                    with _webshop_lock:
                        return env.step(action)

                try:
                    observation, reward, done, info = await asyncio.wait_for(
                        loop.run_in_executor(_webshop_executor, safe_env_step),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    finish_reason = SampleStatus.TASK_ERROR
                    history[-1]["error"] = "env.step() timeout"
                    reward = 0.0
                    break
                except Exception as e:
                    finish_reason = SampleStatus.TASK_ERROR
                    history[-1]["error"] = f"env.step() error: {e}"
                    reward = 0.0
                    break

                history[-1]["reward"] = reward
                history[-1]["done"] = done
                if done:
                    print(f"done with turn = {j}")
                    break
            else:
                finish_reason = SampleStatus.TASK_LIMIT_REACHED
            return TaskOutput(
                status=finish_reason,
                result={
                    "reward": reward,
                    "history": history,
                    "webshop_session_id": episode_session_id,
                },
            )
        finally:
            self._cleanup_episode_session(episode_session_id)

    def calculate_overall(self, results: List[TaskOutput]) -> Dict:
        def factory(key):
            def f(output):
                output = [x for x in output if x]
                if key == "history":
                    return (
                        sum([len(x[key]) for x in output]) / len(output)
                        if len(output) > 0
                        else 0
                    )
                return (
                    sum([x[key] for x in output]) / len(output)
                    if len(output) > 0
                    else 0
                )

            return f

        results = [x.result for x in results if x]

        return {
            "reward": factory("reward")(results),
        }
