
# Guideline to Training an Search Agent Using AReal Framework

This guide provides a complete recipe for training an adanvced search agent based the [AReaL](https://github.com/inclusionAI/AReaL) framework. We'll walk through the entire pipeline from tool integration to trajectory collection, enabling you to train an AI assistant capable of solving complex search tasks with expert-level Search Intelligence.

## Step 1: Search Tool Integration and Environment Configuration

We will begin by integrating necessary tools for the search agent. Specifically, two APIs are used for search and URL access seperately,
1. Serper API for web search 
2. Jina API for URL content retrieval. 

Note that other tools including MCP services can be integrated following a similar approach.

To enable the agent to use tools smoothly, we implement two classes:
1. `OnlineSearchClient` class for sending requests to and receiving responses from tool servers.
2. `SearchToolBox` class that connects the agent with the tools. Specifically, a `step` function is defined to parse agent-generated actions into concrete queries to the `OnlineSearchClient` class.

### 1.1 Implementing a Tool Client

In `OnlineSearchClient` class, we implement a client sending requests to and receiving responses from tool servers. Two functions are supported:
- `search(query: str)`: send a query to search engine and obtain search results
- `access(url: str)`: retrive the content of the webpage at certain URL.

#### 1.1.1 Search
The `search` function processes the input query and returns structured results including document snippets and source URLs.

```python
# AReaL/ASearcher/utils/search_utils.py
class OnlineSearchClient:
    """Core client for handling search operations"""
    
    def search(self, query: str) -> dict:
        """
        Execute a search query and return structured results
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary containing:
            - documents: List of combined title/snippet strings
            - urls: List of corresponding source URLs
        """
        response = requests.post(
            f"{self.serper_server_addr}/search",
            headers=self.serper_headers,
            json=dict(q=query)
        )
        
        data = response.json()
        organic_results = data.get("organic", [])
        
        search_result = {
            "documents": [x.get("title", "") + " " + x.get("snippet", "") 
                        for x in organic_results],
            "urls": [x.get("url", "") for x in organic_results]
        }
        return search_result
```

#### 1.1.2 URL Access 
The `access` function retrieves webpage content using Jina's web reading functionality.

```python
# AReaL/ASearcher/utils/search_utils.py
class OnlineSearchClient:
    """Extended with web access capabilities"""
    
    def access(self, url: str) -> dict:
        """
        Retrieve full content from a specific URL
        
        Args:
            url: Target webpage URL
            
        Returns:
            Dictionary containing:
            - page: Complete webpage content as text
        """
        request_url = f"{self.jina_server_addr}/{url}"
        response = requests.get(request_url, headers=self.jina_headers)
        return {
            "page": response.text
        }
```

### 1.2 Search Environment Setup

#### 1.2.1 Toolbox Initialization
The `SearchToolBox` class parses tool calls from an agent-generated action, uses `OnlineSearchClient` to communicate with tool servers, and computes rewards.

```python
# AReaL/ASearcher/utils/search_tool.py

class SearchToolBox:
    """Orchestrates search operations and manages state"""
    
    def __init__(self):
        """
        Initialize the search environment with:
        - Search client instance
        """
        self.search_client = OnlineSearchClient()
        ...  # Additional environment setup
```

#### 1.2.2 Execute the Action.
The `step` function parses the action of agent into tool calls, execute the tool calls with search client, and calculates the reward.

```python
# AReaL/ASearcher/utils/search_tool.py

class SearchToolBox:
    ...
    
    def step(self, qid_action: Tuple[str, str]) -> dict:
        """
        Process an agent action and return results
        
        Args:
            qid_action: Tuple containing query ID and action string
            
        Returns:
            Dictionary containing:
            - documents/search results
            - URLs (if applicable)
            - Page content (for access actions)
            - Computed score
            - Ground truth reference
        """
        qid, action = qid_action
        result = dict(documents=None, score=None, ground_truth=None, type=None)

        # Search action processing
        if "<search>" in action and "</search>" in action:
            query = extract_between_tags(action, "search")
            response = self.search_client.search(query)
            
            result.update(
                documents=response["documents"],
                urls=response["urls"],
                type="search"
            )
            
        # URL access processing
        elif "<access>" in action and "</access>" in action:
            url = extract_between_tags(action, "access")
            response = self.search_client.access(url)

            result.update(
                page=response["page"],
                type="access"
            )
            
        # Reward computation
        ground_truth = self.id2info[qid]["answer"]
        extracted, score = compute_score(action, ground_truth)
        result.update(
            extracted=extracted,
            score=score,
            ground_truth=ground_truth
        )
        
        return result
```

## Step 2: Constructing a Custom Search Agent

After the tools are ready, we now need to define an agent class. An agent class has the following primary functoinalities:
- Prepare prompts to the LLM
- Extract actions, i.e. tool calls, from the texts generated by the LLM
- Update memory by consuming LLM-generated texts and tool-calling responses.

Correspondingly, this agent is equipped with three key functions:

- `prepare_llm_prompt`: Generate the next prompt based on agent memory.

- `consume_llm_response`: Process the LLM’s output, updates memory, and extracts tool calls.

- `consume_tool_response`: Consume tool execution results and generates follow-up summarization tasks.

### 2.1 Agent Initialization

An agent is initialized with `question`, `memory` and `summary_job_queue`. Here `summary_job_queue` is used to summarize the search results and web contents.

```python
# AReaL/ASearcher/train/search_agent.py

class SearchAgent:

    def __init__(self, question):
        """
        Initialize the search agent with a question.
        
        Args:
            question: Initial question for the agent
        """
        self.question = question
        self.memory = AgentMemory(question=question)  # Stores conversation history
        self.summary_job_queue = queue.Queue(128)  # Buffer for pending tasks
```

### 2.2 LLM Interaction Preparation

The `prepare_llm_prompt` function constructs the input prompt and configures sampling parameters for the LLM. When no search results or web content require processing, the function generates the `llm_prompt` using memory content alone. When there are search results or webpages waiting for summarization, either a search result or webpage content is poped from `summary_job_queue` to append to `llm_prompt`. 

```python
# AReaL/ASearcher/train/search_agent.py

class SearchAgent:
    ...

    def prepare_llm_prompt(self):
        """
        Prepare the input prompt and sampling parameters for LLM generation.
        
        Returns:
            tuple: (prompt_text, sampling_parameters)
        """        
        
        if self.summary_job_queue.empty():
            llm_prompt = self.memory.prepare_prompt()
            sampling_params = dict(stop=["</search>", "</access>", "</answer>"])
        else:
            llm_prompt = self.memory.prepare_prompt()

            job = self.summary_job_queue.get_nowait()
            if job["type"] in ["search_results", "webpage"]:
                # Augment llm_prompt with job information
                llm_prompt += "\n\n" + job["text"] + "\n<think>\n"
                
                # Record the related webpage/search results in memory
                new_record = Record(
                    type=job["type"], 
                    text=job["text"], 
                    short_text=job.get("short_text", job["text"]),
                )
                self.memory.add_record(new_record)
                sampling_params["stop"] = ["</think>"]
                
        return llm_prompt, sampling_params
```

### 2.3 Processing LLM Output  

The `consume_llm_response` method processes the content generated by the LLM, updates the agent's memory, and extracts the tool calls from the LLM's output.  

```python
# AReaL/ASearcher/train/search_agent.py

class SearchAgent:
    ...

    def consume_llm_response(self, resp, completion_text):
        """
        Processes the LLM response, stores the interaction in memory, and extracts potential tool calls.
        
        Args:
            resp: The raw LLM response object.
            completion_text: The decoded text output from the LLM.
            
        Returns:
            list: A list of detected tool calls in the response.
        """
        # Store the LLM interaction in memory
        new_record = Record(
            type="llm_gen",
            text=completion_text,
            input_len=resp.input_len,
            input_tokens=resp.input_tokens,
            output_len=resp.output_len,
            output_tokens=resp.output_tokens,
            output_logprobs=resp.output_logprobs,
            output_versions=resp.output_versions            
        )
        self.memory.add_record(new_record)

        # Extract potential tool calls
        tool_calls = []
        patterns = [
            r'<search>(.*?)</search>', 
            r'<access>(.*?)</access>', 
            r'<answer>(.*?)</answer>'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                match = matches[-1]  # Take the last occurrence
                tool_calls.append(str(pattern.replace('(.*?)', match)))

        return tool_calls
```

### 2.4 Processing Tool Responses

The `consume_tool_response` function processes tool-calling results and create new summarization jobs for the new search results or webpages.

#### 2.4.1 Handling Search Results

```python
# AReaL/ASearcher/train/search_agent.py

class SearchAgent:
    
    def consume_tool_response(self, res: dict, topk: int = 5) -> None:
        """
        Processes tool execution results and generates follow-up summarization tasks.
        
        Args:
            res: Dictionary containing tool response data
            topk: Maximum number of results to process (default: 5)
        """
        if res["type"] == "search":
            # Extract and process top-k search results
            documents = res["documents"][:topk]
            urls = res["urls"][:topk]
            
            formatted_results = process_search_result(documents, urls)
            
            # Create summarization job for search results
            summary_job = {
                "type": "search_results",
                "text": f"<information>{formatted_results}</information>"
            }
            self.summary_job_queue.put_nowait(summary_job)
```

#### 2.4.2 Processing Web Page Content

Webpages can be extremely long, so we divide them into chunks and seperately create multiple summarization tasks for these chunks.

```python
class SearchAgent:
    
    def consume_tool_response(self, res: dict, topk: int = 5) -> None:
        if res["type"] == "search":
        ...
        elif res["type"] == "access":
            # Process webpage content by splitting into manageable chunks
            page_content = res.get("page", "")
            content_chunks = split_page_to_chunks(page_content, chunk_size=10000)
            
            # Create summarization jobs for each content chunk
            for chunk in content_chunks:
                summary_job = {
                    "type": "webpage",
                    "text": f"<information>{chunk}</information>"
                }
                self.summary_job_queue.put_nowait(summary_job)
```


## Step 3: Workflow Definition

Finally, we integrate the environment and agents into a workflow.

### 3.1 Single Trajectory Collection  
We now demonstrate the process of collecting a single trajectory using the predefined `SearchToolBox` and `SearchAgent`. The procedure involves alternately invoking the following sequence of methods until the trajectory is fully completed:  

1. **`agent.prepare_llm_prompt`**: Constructs the LLM prompt based on historical context.  
2. **`engine.agenerate`**: Executes LLM generation using the prepared prompt.  
3. **`agent.consume_llm_response`**: Processes the LLM output, updates agent memory, and extracts potential tool calls.  
4. **`toolbox.step`**: Invokes the search tools and computes the reward.  
5. **`agent.consume_tool_response`**: Handles results of tool calls.  

This iterative cycle continues until the entire trajectory is generated.  

#### 3.1.1 Prepare the prompts & LLM generation
We first invoke `agent.prepare_llm_prompt` and `engine.agenerate`:

```python
# AReaL/ASearcher/train/asearcher.py

class ASearcherWorkflow(RolloutWorkflow):
    """Manages end-to-end search agent operation"""
    
    async def collect_agent_trajectory(self, qid: str, prompt: str, engine) -> Tuple:
        """
        Execute complete search trajectory for a single query
        
        Args:
            qid: Unique query identifier
            prompt: Initial question
            engine: LLM inference engine
            
        Returns:
            Tuple containing:
            - Ground truth answer
            - Final score
            - Complete trajectory
            - Performance statistics
        """
        agent = SearchAgent(prompt)
        toolbox = SearchToolBox()
        score = 0
        ground_truth = None
        traj_rid = uuid.uuid4().hex  # Unique trajectory ID
        
        while agent.num_turns < self.max_turns and not agent.is_finished:
            # Prepare LLM input
            query_prompt, sampling_params = agent.prepare_llm_prompt()
            input_ids = self.tokenizer.encode(query_prompt, add_special_tokens=False)
            
            # Format LLM request
            req = LLMRequest(
                rid=traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(
                    n_samples=1,
                    stop=sampling_params["stop"]
                ),
            )
 
            # Get LLM completion
            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)
            ...
```

#### 3.1.2 Action Execution

After obtaining the LLM output, we invoke `agent.consume_llm_response` to process the model's response. This step updates the agent's memory and extracts any potential tool calls. If tool calls are identified, we then execute them through `toolbox.step` and process the returned results (such as search outputs or web content) using `agent.consume_tool_response`.

```python
# AReaL/ASearcher/train/asearcher.py

class ASearcherWorkflow(RolloutWorkflow):
    ...
    
    async def collect_agent_trajectory(self, qid, prompt, engine):
        ...
        while agent.num_turns < self.max_turns and not agent.is_finished:
            ...
            # Process LLM response
            tool_calls = agent.consume_llm_response(resp, completion_str)[0]

            # Execute action and get results
            if len(tool_calls) > 0:
                res = toolbox.step((qid, tool_calls[0]))
 
                # Update agent memory
                agent.consume_tool_response(res, topk=self.topk)

                # Track performance metrics
                score = res.get("score", score)
                ground_truth = res.get("ground_truth")
        
        # Package final trajectory data
        traj = agent.memory
        stats = agent.memory.logging_stats()
        stats.update(dict(score=score))

        return ground_truth, score, traj, stats
```

### 3.2 Parallel Trajectory Collection

GRPO is used as the RL algorithm for training. A group of $G$ trajectories are collected in parallel for each question. 

```python
# AReaL/ASearcher/train/asearcher.py

class ASearcherWorkflow(RolloutWorkflow):
    ...
    
    async def arun_episode(self, engine, data: dict) -> TensorDict:
        """
        Execute complete training episode for a query
        
        Args:
            engine: LLM inference engine
            data: Contains query information
            
        Returns:
            TensorDict containing all trajectory data
        """
        # Initialize with question prompt
        version = engine.get_version()
        prompt = SEARCH_ACCESS_PROMPT_TEMPLATE.format(question=data["question"])

        # Parallel trajectory collection
        trajs = await asyncio.gather(*[
            self.collect_agent_trajectory(qid, prompt, engine) 
            for _ in range(self.n_trajs)
        ])
```

After the trajectories are collected, they are packed into training format.

```python
class ASearcherWorkflow(RolloutWorkflow):
    ...
    
    async def arun_episode(self, engine, data: dict) -> TensorDict:
        # Parallel trajectory collection
        ...

        # Process and format training data
        results = []
        for i, (_, score, traj, _) in enumerate(trajs):
            for j, record in enumerate(traj.memory):
                if record.type != "llm_gen":
                    continue
                    
                # Convert to training format
                seq = record.input_tokens + record.output_tokens
                logprobs = [0.0] * record.input_len + record.output_logprobs
                loss_mask = [0] * record.input_len + [1] * record.output_len
                versions = [-1] * record.input_len + record.output_versions

                res = dict(
                    input_ids=torch.tensor(seq).unsqueeze(0),
                    loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                    logprobs=torch.tensor(logprobs).unsqueeze(0),
                    versions=torch.tensor(versions).unsqueeze(0),
                    attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                    rewards=torch.tensor([float(score)]),
                )
                results.append(TensorDict(res, batch_size=[1]))

        # Combine all trajectories
        return concat_padded_tensors(results)
```

## Step 4: Training with your Custom Workflow

```python
# AReaL/ASearcher/train/asearcher.py

def main(args):
    ...
    workflow = ASearcherWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        dataset_path=config.train_dataset.path,
        max_turns=config.max_turns,
        n_trajs=config.n_trajs,
        search_client_type=config.search_client_type,
        reward_type=config.reward_type,
        topk=config.topk,
        valid_inst_ratio=config.valid_inst_ratio,
        max_tokens=config.actor.mb_spec.max_tokens_per_mb,
    )

    ...
    data_generator = iter(train_dataloader)
    max_steps = total_epochs * steps_per_epoch
    for global_step in range(start_step, max_steps):

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow, expected_batch_size=worker_batch_size)
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()
        ...
```