"""
Supervisor/Main Agent for routing between sub-agents.
Coordinates the multi-agent system and handles customer queries.
Can be run independently.
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

from .config import LLM_CONFIG, SUPERVISOR_PROMPT
from .state import State
from .searcher_agent import create_searcher_agent
from .summary_agent import create_summary_agent

def create_supervisor(state: State, config: RunnableConfig, store: BaseStore):
    """
    Create and compile the supervisor agent.
    
    Returns:
        langgraph.graph: Compiled supervisor agent graph.
    """
    # Initialize the LLM
    llm = ChatOpenAI(**LLM_CONFIG)
    
    # Initialize memory components
    checkpointer = MemorySaver()
    
    in_memory_store = InMemoryStore()

    # Create sub-agents
    searcher_agent = create_searcher_agent()
    summary_agent = create_summary_agent()
    
    # Import the supervisor creator to avoid name collision
    from langgraph_supervisor import create_supervisor as create_supervisor_workflow

    supervisor_prebuilt_workflow = create_supervisor_workflow(
            agents=[searcher_agent, summary_agent],
            output_mode="last_message",
            model=llm,
            prompt=SUPERVISOR_PROMPT,
            state_schema=State
        )
        
    supervisor_prebuilt = supervisor_prebuilt_workflow.compile(
            name="supervisor",
            checkpointer=checkpointer,
            store=in_memory_store
        )
    
    return supervisor_prebuilt


if __name__ == "__main__":
    """Test the supervisor independently."""
    import uuid
    from langchain_core.messages import HumanMessage
    
    # Create the supervisor
    supervisor = create_supervisor()
    
    thread_id = uuid.uuid4()
    question = "请帮我分析一下当前的股票市场行情，并给出投资建议。"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Testing Supervisor Agent (Invoice Query)")
    print(f"Thread ID: {thread_id}")
    print(f"Question: {question}\n")
    
    result = supervisor.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    
    print("Response:")
    for message in result["messages"]:
        message.pretty_print()
