from agent.search_r1 import SearchR1Agent  
from agent.asearcher_reasoning import AsearcherReasoningAgent
from agent.asearcher import AsearcherAgent


AGENT_CATEGORY = {
    "search-r1": SearchR1Agent,
    "asearcher-reasoning": AsearcherReasoningAgent,
    "asearcher": AsearcherAgent,
}

def make_agent(agent_type):
    return AGENT_CATEGORY[agent_type]()