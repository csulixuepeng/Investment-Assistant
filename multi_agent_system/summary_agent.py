"""LLM node for reasoning and tool selection."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .config import LLM_CONFIG, SUMMARY_ASSISTANT_PROMPT_TEMPLATE
from .state import State
from .utils import show_graph


def create_summary_agent():
    # Initialize the LLM
    llm = ChatOpenAI(**LLM_CONFIG)
    
    def summary_assistant(state: State, config: RunnableConfig):
        """LLM node for reasoning and tool selection."""
        memory = "None" 
        if "loaded_memory" in state: 
            memory = state["loaded_memory"]

        summary_assistant_prompt = SUMMARY_ASSISTANT_PROMPT_TEMPLATE
        
        response = llm.invoke(
            [SystemMessage(summary_assistant_prompt)] + state["messages"]
        )
        
        return {"messages": [response]}
    
    # Build the graph
    summary_workflow = StateGraph(State)
    
    # Add nodes
    summary_workflow.add_node("summary_assistant", summary_assistant)    
    # Add edges
    summary_workflow.add_edge(START, "summary_assistant")
    summary_workflow.add_edge("summary_assistant", END)    
    # Initialize memory components
    checkpointer = MemorySaver()
    in_memory_store = InMemoryStore()
    
    # Compile the graph
    summary_subagent = summary_workflow.compile(
        name="summary_subagent",
        checkpointer=checkpointer,
        store=in_memory_store
    )
    
    return summary_subagent


if __name__ == "__main__":
    import uuid
    from langchain_core.messages import HumanMessage
    
    # Create the agent
    summary_agent = create_summary_agent()
    
    # Test with a sample query
    thread_id = uuid.uuid4()
    question = "Can you provide a summary of the following Python code that implements a binary search algorithm?"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Thread ID: {thread_id}")
    print(f"Question: {question}\n")
    
    # Invoke the agent
    result = summary_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    
    # Print results
    print("Response:")
    for message in result["messages"]:
        message.pretty_print()
