import re
import time
import random
import uuid
import json
import copy
import asyncio, aiohttp
import requests
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional

class ConstructQAPrompts:
    # base & link qa construct
    base_qa='''You are an autonomous agent for constructing general-domain QAs. The following related material are extracted from a webpage. Based on the given material, propose a simple question. Please ensure that the question is clear, solvable, and has an unique answer. The question should not involve complex information but should require some superior knowledge to find out the answer.

The output should be in json format:
```json
{{
    "question": the proposed question,
    "answer": answer to the proposed question,
    "statement": a declarative sentence for the fact coverred in the proposed question-answer pair.
}}
```

# Material
{content}
'''
    link_qa='''You are an autonomous agent for constructing general-domain QAs. Given the information of two entities, please propose a question where the answer is {entityA} and the question context is about {entityB}. Please ensure that the question is clear, solvable, and has an unique answer. The question should not involve complex information but should require some superior knowledge to find out the answer.

Information About {entityA}:
```txt
{contentA}
```

Information About {entityB}:
```txt
{contentB}
```

The output should be in json format:
```json
{{
    "question": the proposed question,
    "answer": answer to the proposed question,
    "statement": a declarative sentence for the fact coverred in the proposed question-answer pair.
}}
```
'''

    # compose qa
    compose_qa='''You are an autonomous agent for constructing general-domain QAs. Now given two questions, combine these questions into one. Specially, the answer to the second question is related to some entity in the first question. To make the combined question challenging, you need to remove information about the answer of the second question and ensure the answer of the combined question remains the same as the answer of the first question. Please ensure that the combined question is clear, solvable, and has an unique answer.

The first Question:
```
{questionA}
```

The second Question:
```
{questionB}
```

Relevant statements:
```
{statements}
```

The output should be in json format:
```json
{{
    "question": the proposed question,
    "answer": the answer
    "note": one short desription of how the two questions are combined
}}
```
'''
    compose_qa_by_statement='''You are an autonomous agent for constructing general-domain QAs. Now given a question-answer pair and an entity, combine these the entity into the question-answer pair. Specially, the entity is accompanied with a statement that connects the entity with the question-answer pair. To make the combined question challenging, you need to remove information about that connects the entity and the question-answer pair and ensure the answer of the combined question remains the same as the answer of the first question. Please ensure that the combined question is clear, solvable, and has an unique answer.

The question-answer pair:
```
{question}
```

The entity and the related statement:
```
{entity}
```

Relevant statements supporting the question-answer pair:
```
{statements}
```

The output should be in json format:
```json
{{
    "question": the proposed question,
    "answer": the answer
    "note": one short desription of how the entity is combined into the question are combined
}}
```
'''

    # action
    action = '''You are an autonomous agent for constructing general-domain QAs. Now given the current QA and relevant information. Choose one of the following action to make the question more challenging.

The current question-answer pair:
{question}

You can choose one action from the following types:
{actions}

'''

    SELECT='''SELECT: select one entity from the relevant entity list. Once such an entity is selected, an external tool will improve the difficulty of the question by replacing information about this entity in the question with sub-questions that take this entity as the answer.

If you choose SELECT, the output should be in json format:
```json
{
    "action": "SELECT",
    "target": url of the selected entity. note that you should only select the entity from the relevant entity list of the question and make sure the url exactly match the url in the relevant entity list.
    "note": a short description of the rationale behind the selection
}
```'''

    FUZZ='''FUZZ: fuzz 1 places of information in the question to make the question more challenging. Note that if you choose FUZZ, you should make sure the resulted question is still clear and has an unique answer as the original one. You should choose FUZZ only when you find certain pieces of information could clearly point to the correct answer without extensive research to find relevant information.

If you choose FUZZ, the output should be in json format:
```json
{
    "action": "FUZZ",
    "question": the modified question after the FUZZ operation 
    "note": a short description of why and how the FUZZ operation happens
}
```'''

    EXIT='''EXIT: exit when you find the question safisfying all the following criteria:
- the question requires extensive research to find the correct answer
- information provided in the question is vague enough such that no single piece of information could clearly point to the correct answer. 

If you choose EXIT, the output should be in json format:
```json
{
    "action": "EXIT",
    "note": a short description of why you choose to exit
}
```'''

    BRAINSTORM='''BRAINSTORM: brainstorm a list of potential entities that may have connection with the current question-answer pair. The brainstormed entities would then be merged into the question to make the question more challenging. The connection could be weak, e.g. year, geological area, organizations, some people, and so on. You also need to ensure the related fact of the brainstormed entities is real.
    
If you choose BRAINSTORM, the output should be in json format:
```json
{
    "action": "BRAINSTORM",
    "entities": [
        {
            "name": name of the entity,
            "wiki_url": the link to the entity on wikipedia,
            "statement": a statement of the fact about the entity that connects it to the current question-answer
        }, // the first entity
        // the second entity,
        ...
    ]
}
```'''

    # select entity neighbor
    select_neighbor = None # this is currently done by random

    # utils
    summarize_webpage = '''Given a webpage, summarize the content of this webpage.
    
Webpage content:
```markdown
{content}
```

The output should enclose the title and summary in <title> </title> and <summary> </summary> tags respectively.

'''
    extract_information_points = '''Given a webpage, extract all information points in this webpage.
    
Webpage content:
```markdown
{content}
```

The output should be a list of strings in json format:
```json
[
    information point 1,
    information point 2,
    ...
]
```

'''

    # quality check
    check_info_cover = '''Check whether the information of a statement is fully covered by prior statements.

Prior statements:
```txt
{prior}
```

Statement to be checked:
```txt
{current}
```

You should reply "yes" or "no" indicating whetherthe information of the statement is fully covered by prior statements. You should think step-by-step first before the final judgement in json format:

# Analysis

// your analysis

# Final Judgement
```json
{{
    "judgement": "yes" or "no"
}}

'''
    check_alternative_ans = '''Determine whether the predicted answer is also correct to the question. Specially, the predicted answer is different from the ground-truth answer, and you should check whether the predicted answer fits with all constraints in the question and is also a correct answer to the question.
    
Question: {question}

Ground-truth answer: {gt_answer}

Facts supporting the ground-truth answer:
```txt
{statements}
```

Predicted answer: {pred_answer}

```json
{{
    "judgement": "yes" or "no"
}}
```
'''
    qa_valid_check = '''Check the validity of this question-answer pair given its relevant information.

The question is valid is and only if:
1. the question is not a simple concatenation of two or more questions
2. the provided answer is the only correct answer to the question
3. the question has an unique answer
4. the question can be solved based on the relevant statements

The question-answer pair and the relevant information:
{question}

You should reply "yes" or "no" indicating the validity of the question-answer pair. You should think step-by-step first before the final judgement in json format:

# Analysis

// your analysis

# Final Judgement
```json
{{
    "judgement": "yes" or "no"
}}
```
'''
    direct_gen_check = """Answer the following question and put your answer within <answer> </answer> tags. \n\n{question}"""
    llm_judge =  """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {gt_answer}

Predicted Answer: {pred_answer}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""


class AgentMemory:
    '''store the memory, including:
    1. the current question & answer 
    2. the basic statements
    3. the relevent entities and the relevant links
    4. the edit history
    '''

    def __init__(self):
        self.qa = dict(question=None, answer=None)
        self.statements = []
        self.relevant = []
        self.edit_history = []
        self.qa_history = []
    
    def repr(self):
        relevant = '\n'.join([f'- [{e.name}]({e.url})' for e in self.relevant])
        statements = '\n'.join(self.statements)
        ret = f"""
Question: {self.qa['question']}
Answer: {self.qa['answer']}

Relevant Statements:
```txt
{statements}
```

Relevant Entity List:
``` txt
{relevant}
```
"""
        return ret

    def statements_repr(self, additional=None):
        return '\n'.join(self.statements + (additional or []))

    def dict(self):
        return dict(
            qa=self.qa,
            relevant=[e.dict() for e in self.relevant],
            statements=self.statements,
            edit_history=self.edit_history,
            qa_history=self.qa_history,
        )

class WebPage:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.summary = None
        self.information_points = None
        self.relevant_links = None
    
    def repr(self):
        ret = f"Title: {self.name}\n URL: {self.url}\n\n# Summary\n\n{self.summary}\n\n# Information Points\n\n{self.information_points}"
        return ret
    
    def information_points_repr(self):
        return self.information_points
    
    def dict(self):
        return dict(
            name=self.name,
            url=self.url,
            summary=self.summary,
            information_points=self.information_points,
            relevant_links=self.relevant_links,
        )

def normalize_url(url):    
    if "index.php/" in url:
        url = url.replace("index.php/", "index.php?title=")
    if "/wiki/" in url:
        url = url.replace("/wiki/", "/w/index.php?title=")
    if "_" in url:
        url = url.replace("_", "%20")
    return url

def find_page(pages, url):
    url = normalize_url(url)
    return pages.get(url, None)

def exists_page(pages, url):
    url = normalize_url(url)
    return url in pages

class ConstructQAAgent:
    def __init__(self, all_links, pages, search_client, max_turns=16):
        self.all_links = all_links
        self.pages = pages
        self.all_urls = list(all_links.keys())
        self.search_client = search_client
        self.max_turns = max_turns
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen__Qwen2.5-32B")

    
    async def call_llm(self, prompt, client):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) 
        print("PROMPT:", [prompt[:1000]])
        max_new_tokens = 32000 - self.tokenizer([prompt], return_length=True)["length"][0]
        sampling_kwargs = dict(
            temperature=0.8,
            # sseed=args.seed,
            top_p=0.95,
            top_k=1000,
            max_new_tokens=max_new_tokens,
            n=1,
            stop_token_ids=[151645, 151643]
        )
        try:
            output = await client.async_generate(prompt, sampling_kwargs)
        except ValueError as e:
            print("ValueError when calling llm")
            raise e
        # print(prompt)
        print(output["text"])
        return output["text"]
        
    async def extract_webpage(self, url, client) -> WebPage:
        print(f"[INFO] extract information from webpage: {url}", flush=True)
        name = url
        webpage = WebPage(name, url)
        content = find_page(self.pages, url)["contents"] # await self.search_client.access_async([url])
        assert content is not None
        if isinstance(content, dict) and "page" in content:
            content = content["page"]
        title_summary = await self.call_llm(ConstructQAPrompts.summarize_webpage.format(content=content[:50000]), client)
        information_points = await self.call_llm(ConstructQAPrompts.extract_information_points.format(content=content[:50000]), client)

        title, summary, info_pts = None, None, None

        if "<title>"in title_summary and "</title>" in title_summary:
            title = title_summary.split("<title>")[-1].split("</title>")[0].strip()
            name = title
        if "<summary>" in title_summary and "</summary>" in title_summary:
            summary = title_summary.split("<summary>")[-1].split("</summary>")[0].strip()
        if "```json" in information_points and "```" in information_points.split("```json")[-1]:
            info_pts = information_points.split("```json")[-1].split("```")[0].strip()
        
        webpage.name = name
        webpage.summary = summary
        webpage.information_points = info_pts
        webpage.relevant_links = self.all_links[normalize_url(url)]["links"] + self.all_links[normalize_url(url)]["in_links"]

        return webpage

    async def construct_base_qa(self, entity, client):
        prompt = ConstructQAPrompts.base_qa.format(content=entity.repr())
        text = await self.call_llm(prompt, client)
        base_qa = json.loads(text.split("```json")[-1].split("```")[0].strip())
        return base_qa
    
    async def choose_action(self, state, client, ready_to_exit=False):
        print(f"choose action @ state\n{state}", flush=True)
        actions = [ConstructQAPrompts.FUZZ, ConstructQAPrompts.SELECT] # , ConstructQAPrompts.BRAINSTORM]
        # actions = [ConstructQAPrompts.BRAINSTORM]
        random.shuffle(actions)
        if ready_to_exit:
            actions.append(ConstructQAPrompts.EXIT)
        prompt = ConstructQAPrompts.action.format(question=state, actions=actions)
        text = await self.call_llm(prompt, client)
        action = json.loads(text.split("```json")[-1].split("```")[0].strip())
        assert "action" in action
        if action["action"] == "SELECT":
            assert "target" in action and "note" in action
        elif action["action"] == "FUZZ":
            assert "question" in action
            assert "note" in action
        elif action["action"] == "EXIT":
            assert "note" in action
        elif action["action"] == "BRAINSTORM":
            assert "entities" in action
            assert all([isinstance(e, dict) and 'name' in e and 'wiki_url' in e and 'statement' in e for e in action['entities']])
        assert action["action"] in ["SELECT", "FUZZ", "EXIT", "BRAINSTORM"]
        return action
    
    async def construct_link_qa(self, targetA, targetB, client):
        prompt = ConstructQAPrompts.link_qa.format(entityA=targetA.name, entityB=targetB.name, contentA=targetA.repr(), contentB=targetB.repr())
        text = await self.call_llm(prompt, client)
        link_qa = json.loads(text.split("```json")[-1].split("```")[0].strip())
        assert isinstance(link_qa, dict)
        assert ("question" in link_qa and "answer" in link_qa and "statement" in link_qa), (link_qa.keys())
        return link_qa
    
    async def check_info_cover(self, statement, prior_statements, client):
        prompt = ConstructQAPrompts.check_info_cover.format(prior=prior_statements, current=statement)
        text = await self.call_llm(prompt, client)
        result = json.loads(text.split("```json")[-1].split("```")[0].strip())
        assert isinstance(result, dict)
        assert "judgement" in result
        return result["judgement"] == "yes"
    
    async def combine_qa(self, questiona, questionb, memory, client):
        prompt = ConstructQAPrompts.compose_qa.format(questionA=json.dumps(dict(question=questiona["question"], answer=questiona["answer"])), questionB=json.dumps(dict(question=questionb["question"], answer=questionb["answer"])), statements=memory.statements_repr(additional=[questionb["statement"]]))
        text = await self.call_llm(prompt, client)
        combine_qa = json.loads(text.split("```json")[-1].split("```")[0].strip())
        print(f"[DEBUG] combine qa: {combine_qa}")
        assert isinstance(combine_qa, dict)
        assert ("question" in combine_qa and "answer" in combine_qa and "note" in combine_qa)
        assert (combine_qa["answer"] == questiona["answer"]), (combine_qa)
        return combine_qa
    
    async def combine_qa_by_statement(self, question, entity, memory, client):
        
        prompt = ConstructQAPrompts.compose_qa_by_statement.format(question=json.dumps(dict(question=question["question"], answer=question["answer"])), entity=json.dumps(dict(name=entity["name"], statement=entity["statement"])), statements=memory.statements_repr())
        text = await self.call_llm(prompt, client)
        combine_qa = json.loads(text.split("```json")[-1].split("```")[0].strip())
        print(f"[DEBUG] combine qa by statement: {combine_qa}")
        assert isinstance(combine_qa, dict)
        assert ("question" in combine_qa and "answer" in combine_qa and "note" in combine_qa)
        assert (combine_qa["answer"] == question["answer"]), (combine_qa)
        return combine_qa
    
    async def check_qa_valid(self, state, client):
        prompt = ConstructQAPrompts.qa_valid_check.format(question=state)
        text = await self.call_llm(prompt, client)
        result = json.loads(text.split("```json")[-1].split("```")[0].strip())
        print(f"[DEBUG] llm judge result: {result}")
        assert isinstance(result, dict) and "judgement" in result
        return 'yes' in result["judgement"]
    
    async def direct_generate(self, question, client, n=1):
        prompt = ConstructQAPrompts.direct_gen_check.format(question=question)
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) 
        max_new_tokens = 32000 - self.tokenizer([prompt], return_length=True)["length"][0]
        sampling_kwargs = dict(
            temperature=0.6,
            # sseed=args.seed,
            top_p=0.95,
            top_k=1000,
            max_new_tokens=max_new_tokens,
            n=n,
            stop_token_ids=[151645, 151643]
        )
        print("PROMPT:", [prompt[:1000]], flush=True)
        try:
            output = await client.async_generate(prompt, sampling_kwargs)
        except ValueError as e:
            print("ValueError when calling llm", flush=True)
            raise e
        texts = [o["text"] for o in output]
        print("# of texts", len(texts), flush=True)
        assert len(texts) == n
        answers = [None for i in range(n)]
        for i in range(n):
            if "<answer>" in texts[i] and "</answer>" in texts[i]:
                answers[i] = texts[i].split("<answer>")[-1].split("</answer>")[0].strip()
        return answers
    
    async def llm_judge_answer(self, question, answers, gt_answer, client):
        corrects = []
        for ans in answers:
            if ans is None:
                corrects.append(0)
            else:
                prompt = ConstructQAPrompts.llm_judge.format(question=question, gt_answer=gt_answer, pred_answer=str(ans))
                text = await self.call_llm(prompt, client)
                corrects.append("Correct" in text)
        return corrects
    
    async def check_alternative_answer(self, question, gt_answer, pred_answer, statements, client):
        prompt = ConstructQAPrompts.check_alternative_ans.format(question=question, gt_answer=gt_answer, pred_answer=pred_answer, statements=statements)
        text = await self.call_llm(prompt, client)
        # result = json.loads(text.split("```json")[-1].split("```")[0].strip())
        print(f"[DEBUG] check alternative answer result: {text}")
        # assert isinstance(result, dict) and "judgement" in result
        return 'yes' in text.split('</think>')[-1]
        
    async def generate(self, semaphore, reasoning_client, instruct_client, save_path):
        async with semaphore:
            # sample a root entity
            root_url = random.choice(self.all_urls)
            memory = AgentMemory()
            uid = str(uuid.uuid4())
            root_entity: WebPage = await self.extract_webpage(root_url, instruct_client)
            memory.relevant.append(root_entity)
            memory.uid = uid

            print("start generting qa")

            try:
                base_qa = await self.construct_base_qa(root_entity, reasoning_client)
            except:
                print("[ERROR] generate base qa failed")
                return None
            if base_qa is None or not (isinstance(base_qa, dict) and "question" in base_qa and "answer" in base_qa and "statement" in base_qa):
                print("[ERROR] generate base qa failed")
                return None
            
            memory.qa["question"] = base_qa["question"]
            memory.qa["answer"] = base_qa["answer"]
            memory.statements.append(base_qa["statement"])
            memory.qa_history.append(base_qa)
            memory.edit_history.append(f"Create a base question from entity: {root_entity.name}\nQuestion: {base_qa['question']}\nAnswer: {base_qa['answer']}")

            print(memory.edit_history[-1])

            ready_to_exit = False

            action_stats = defaultdict(int)
        
            for turn in range(self.max_turns):
                print(f"Turn: {turn}")
                state = memory.repr()
                if turn == 0:
                    action = dict(action="none")
                else:
                    try:
                        action = await self.choose_action(state, reasoning_client, ready_to_exit=ready_to_exit)
                    except:
                        print(f"[WARNING] generate action failed at turn {turn}")
                        continue
                print("History:\n\n{}".format("\n".join(memory.edit_history)))
                
                q_new = None
                memory_new = copy.deepcopy(memory)
                memory_new.edit_history.append(f"Action: {action['action']}. Note: {action.get('note', None)}")
                print(action)
                action_stats[action["action"]] += 1
                if action["action"] == "FUZZ":
                    q_new = action["question"]
                    memory_new.edit_history.append(f"FUZZ operation modifies the question to: {q_new}")
                elif action["action"] == "EXIT":
                    print("[INFO] question generation is done, exiting")
                    break
                elif action["action"] == "none":
                    assert turn == 0
                    q_new = base_qa["question"]
                elif action["action"] == "SELECT":
                    # find the target entity
                    target = None
                    for e in memory.relevant:
                        if e.url == action["target"]:
                            target = e
                            break
                    if target is None:
                        print(f"[WARNING] target url {action['target']} is not found. the full list is {[e.url for e in memory.relevant]}. skip")
                        continue
                    candidates = [normalize_url(l) for l in target.relevant_links]
                    exist_links = [normalize_url(e.url) for e in memory.relevant]
                    candidates = [c for c in candidates if c not in exist_links]
                    if len(candidates) == 0:
                        print(f"[WARNING] {target.url} has no suitable links. skip")
                        continue
                    neighbor_url =  random.choice(candidates)
                    print(f"SELECT -> Neighbor: {target.url} -> {neighbor_url}")
                    targetA = target
                    targetB: WebPage = await self.extract_webpage(neighbor_url, instruct_client)

                    # propose basic link QA
                    try:
                        link_qa: Dict = await self.construct_link_qa(targetA, targetB, reasoning_client)
                    except:
                        print(f"[WARNING] error when constructing link qa at turn {turn}. skip")
                        continue
                        
                    # check for duplicate statement
                    try:
                        duplicate: bool = await self.check_info_cover(link_qa["statement"], memory_new.statements_repr(), reasoning_client)
                    except:
                        print(f"[WARNING] error when checkint duplicate statement at turn {turn}. skip")
                        continue
                    
                    if duplicate:
                        print(f"[WARNING] the statement of the constructed link qa at turn {turn} is duplicated. skip")
                        continue
                    
                    # combine the link qa into the base qa
                    # if True:
                    try:
                        combine_qa = await self.combine_qa(memory.qa, link_qa, memory, reasoning_client)
                        q_new = combine_qa["question"]
                    except:
                        print(f"[WARNING] error when composing qa at turn {turn}. skip")
                        continue
                    memory_new.relevant.append(targetB)
                    memory_new.statements.append(link_qa["statement"])
                    memory_new.edit_history.append(f"Select '{targetA.name}' and its relevant entity '{targetB.name}'")
                    memory_new.edit_history.append(f"Construct a basic QA for  '{targetA.name}' and its relevant entity '{targetB.name}': Q={link_qa['question']} A={link_qa['answer']}")
                    memory_new.edit_history.append(f"Combine the created basic QA into the base QA. The new question is '{q_new}'. Answer should be '{memory.qa['answer']}'")
                elif action["action"] == "BRAINSTORM":
                    # check the validity of the wiki urls
                    entities = action["entities"]
                    valid_entities = []
                    for e in entities:
                        ex = exists_page(self.pages, e["wiki_url"])
                        ex = ex and not any([normalize_url(e_.url) == normalize_url(e["wiki_url"]) for e_ in memory.relevant])
                        if not ex:
                            print(f"[WARNING] url not found: {json.dumps(e)}")
                        else:
                            valid_entities.append(e)
                    if len(valid_entities) == 0:
                        print(f"[WARNING] no valid entities found at turn {turn}")
                        await asyncio.sleep(10)
                        continue
                    success = False
                    for e in valid_entities:
                        url = e["wiki_url"]
                        memory_e = copy.deepcopy(memory_new)

                        target: WebPage = await self.extract_webpage(url, instruct_client)

                        # check validity of the statement
                        try:
                            info_in_target: bool = await self.check_info_cover(e["statement"], target.information_points_repr(), reasoning_client)
                            info_in_current: bool = await self.check_info_cover(e["statement"], memory.statements_repr(), reasoning_client)
                            info_in_both: bool = await self.check_info_cover(e["statement"], memory.statements_repr() + "\n" + target.information_points_repr(), reasoning_client)
                        except:
                            print(f"[WARNING] error when checkint info cover of brainstorm entity statement at turn {turn}. skip")
                            continue
                            
                        if not info_in_both:
                            print(f"[WARNING] the brainstormed statement is not valid at turn {turn}. unsupported information is found: {json.dumps(e)}")
                            # await asyncio.sleep(10)
                            continue
                            
                        if info_in_current:
                            # no need to combine the statement into the question, only add the entity
                            memory_e.relevant.append(target)
                            memory_e.edit_history.append(f"Brainstorm '{e['name']}' w. statement: {e['statement']} (Note: information already coverred)")
                            q_new = memory_e.qa["question"]
                        else:
                            # combine the statment with the question
                            try:
                                combine_qa = await self.combine_qa_by_statement(memory.qa, e, memory, reasoning_client)
                                q_new = combine_qa["question"]
                            except:
                                print(f"[WARNING] error when composing qa at turn {turn}. skip")
                                continue
                            memory_e.relevant.append(target)
                            memory_e.statements.append(e["statement"])
                            memory_e.edit_history.append(f"Brainstorm '{e['name']}' w. statement: {e['statement']}")
                            memory_e.edit_history.append(f"Combine the created basic QA into the base QA. The new question is '{q_new}'. Answer should be '{memory.qa['answer']}'")
                        success = True
                        memory_new = memory_e
                        break
                    if not success:
                        print(f"[WARNING] brainstorm failed at turn {turn}.")
                        continue

                print(f"New Question at turn {turn}:\n{q_new}")
                memory_new.qa["question"] = q_new

                # check validity
                valid = False
                try:
                    valid = await self.check_qa_valid(memory_new.repr(), reasoning_client)
                except:
                    pass
                if not valid:
                    print(f"[WARNING] the constructed qa is invalid at turn {turn}.")
                    continue

                # direct generation check
                try:
                    answers = await self.direct_generate(q_new, reasoning_client, n=8)
                except:
                    print(f"[WARNING] direct llm generation failed")
                    continue

                try:
                    corrects = await self.llm_judge_answer(q_new, answers, memory.qa["answer"], instruct_client)
                except:
                    print(f"[WARNING] llm judge failed")
                    continue
            
                # check for alternative answers
                is_alternative = False
                for pred_ans, correct in zip(answers, corrects):
                    if not correct:
                        try:
                            is_alternative = await self.check_alternative_answer(q_new, memory.qa["answer"], pred_ans, memory.statements_repr(), reasoning_client)
                        except:
                            print(f"[WARNING] checking alternative answer failed for '{pred_ans}'")
                        if is_alternative:
                            break
                if is_alternative:
                    print(f"[WARNING] {pred_ans} is an alternative answer to '{q_new}' besides {memory.qa['answer']}. skip")
                    continue

                
                memory = memory_new
                print("[INFO] direct gen answers & corrects: {}".format([(a,c) for a, c in zip(answers, corrects)]))
                memory.qa_history.append(dict(
                    question=q_new,
                    answer=memory.qa["answer"],
                    direct_gen_acc="{}/{}".format(sum(corrects), len(corrects))
                ))
                memory.edit_history.append("[INFO] QwQ Agent direct gen accuracy: {}/{}".format(sum(corrects), len(corrects)))
                memory.edit_history.append("[INFO] QwQ Agent direct gen answers & corrects: {}".format([(a,c) for a, c in zip(answers, corrects)]))
                if not any(corrects):
                    ready_to_exit=True
                    print(f"[INFO] current question at turn {turn} is not correct out of n direct generations")
                    memory.edit_history.append(f"[INFO] current question at turn {turn} is not correct out of n direct generations")
                    memory.qa_history.append(copy.deepcopy(memory.qa))
                print(f"Action stats: {action_stats}", flush=True)
            with open(f"{save_path}/{uid}.jsonl", "w") as f:
                _=f.write(json.dumps(memory.dict()))
            return memory
            
async def main(agent, reasoning_client, instruct_client, save_path):
    semaphore = asyncio.Semaphore(128)
    await instruct_client.__aenter__()
    tasks = [agent.generate(semaphore, reasoning_client, instruct_client, save_path) for i in range(1024*8)]
    results = await asyncio.gather(*tasks)
    await instruct_client.__aexit__()

class SGLangAPIClient:
    def __init__(self, key="qwen2.5-72b-inst"):
        self.base_url  = None
        self.init_llm_client(key)
    
    async def __aenter__(self):
        AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(
            total=6 * 60 * 60,
            connect=300,
        )
        conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, force_close=True)
        self.session = aiohttp.ClientSession(
            timeout=AIOHTTP_TIMEOUT,
            connector=conn,
            read_bufsize=1024 * 1024 * 10,
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def init_llm_client(self, key):
        self.server_list = self.get_llm_server(key)
        random.shuffle(self.server_list)
        for server_addr in self.server_list:
            url = f"http://{server_addr}/get_model_info"
            try:
                res = requests.get(
                    url, timeout=5, headers={}
                )
                print(url, res.status_code, res.text)
                assert res.status_code == 200, f"{res=}, {res.text=}"
                success = True
                self.base_url = f"http://{server_addr}"
                break
            except (AssertionError, requests.exceptions.RequestException):
                print(f"[WARNING] server {server_addr} is not in use")
                pass
        assert self.base_url is not None
        print(f"Connected to {self.base_url}")

    def get_llm_server(self, key="qwen2.5-72b-inst"):
        # get server list
        raise NotImplementedError("Add the server for different LLMs/LRMs here")
        return server_list

    def remove_prefix(self, text: str, prefix: str) -> str:
        return text[len(prefix) :] if text.startswith(prefix) else text

    async def async_generate(self, prompt, sampling_kwargs):
        n = sampling_kwargs.get("n", 1)
        payload = {
            "text": prompt,
            "sampling_params": sampling_kwargs,
            "stream": False,
        }
        outputs = [None for _ in range(n)]
        output_idx = 0
        st=time.time()
        async with self.session.post(url=f"{self.base_url}/generate", json=payload) as response:
            response.raise_for_status()
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = self.remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                latency = time.perf_counter() - st
                if chunk == "[DONE]":
                    pass
                else:
                    datas = json.loads(chunk)
                    if not isinstance(datas, list):
                        datas = [datas]
                    for data in datas:
                        print("[DEBUG] sglang return data", data.keys(), flush=True)
                        outputs[output_idx] = data["text"]
                        output_idx +=1
        if len(outputs) == 1:
            outputs = outputs[0]
        return {"text": outputs}



if __name__ == "__main__":
    from llm_utils import get_sglang_llm
    from search_utils import AsyncOnlineSearchClient
    from argparse import Namespace
    import asyncio
    import tqdm
    
    pages_path = "path_to_wiki2018_pages_file"
    links_path = "path_to_wiki2018_links_file"
    save_path = ""

    # load pages
    pages = []
    for ff in tqdm.tqdm(open(pages_path,"r")):
        pages.append(json.loads(ff))
    pages = {page["url"]: page  for page in pages}

    # all links
    all_links = json.load(open(links_path, "r"))

    # llm client
    llm_client = SGLangAPIClient("QwQ-32B")
    instruct_client = SGLangAPIClient("Qwen2.5-72B-Instruct")

    # search server is not used
    search_client = None 

    agent = ConstructQAAgent(all_links, pages, search_client)

    asyncio.run(main(agent, llm_client, instruct_client, save_path))