"""LLM node for reasoning and tool selection."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .config import LLM_CONFIG, SEARCHER_ASSISTANT_PROMPT_TEMPLATE
from .state import State
from .utils import show_graph
from .searcher_tool import get_searcher_tools


def create_searcher_agent():
    # Initialize the LLM
    llm = ChatOpenAI(**LLM_CONFIG)
    
    searcher_tools = get_searcher_tools()
    
    # Bind tools to LLM
    llm_with_searcher_tools = llm.bind_tools(searcher_tools)
    
    # Create ToolNode for executing tools
    searcher_tool_node = ToolNode(searcher_tools)
    
    def searcher_assistant(state: State, config: RunnableConfig):
        """LLM node for reasoning and tool selection."""
        memory = "None" 
        if "loaded_memory" in state: 
            memory = state["loaded_memory"]

        searcher_assistant_prompt = SEARCHER_ASSISTANT_PROMPT_TEMPLATE
        
        response = llm_with_searcher_tools.invoke(
            [SystemMessage(searcher_assistant_prompt)] + state["messages"]
        )
        
        return {"messages": [response]}
    
    # Define conditional edge function
    def should_continue(state: State, config: RunnableConfig):
        """Determine whether to continue tool execution or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
    
    # Build the graph
    searcher_workflow = StateGraph(State)
    
    # Add nodes
    searcher_workflow.add_node("searcher_assistant", searcher_assistant)
    searcher_workflow.add_node("searcher_tool_node", searcher_tool_node)
    
    # Add edges
    searcher_workflow.add_edge(START, "searcher_assistant")
    searcher_workflow.add_conditional_edges(
        "searcher_assistant",
        should_continue,
        {
            "continue": "searcher_tool_node",
            "end": END,
        },
    )
    searcher_workflow.add_edge("searcher_tool_node", "searcher_assistant")
    
    # Initialize memory components
    checkpointer = MemorySaver()
    in_memory_store = InMemoryStore()
    
    # Compile the graph
    searcher_subagent = searcher_workflow.compile(
        name="searcher_subagent",
        checkpointer=checkpointer,
        store=in_memory_store
    )
    
    return searcher_subagent


if __name__ == "__main__":
    import uuid
    from langchain_core.messages import HumanMessage
    
    # Create the agent
    searcher_agent = create_searcher_agent()
    
    # Test with a sample query
    thread_id = uuid.uuid4()
    question = "Who is Leo Messi??"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Thread ID: {thread_id}")
    print(f"Question: {question}\n")
    
    # Invoke the agent
    result = searcher_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    
    # Print results
    print("Response:")
    for message in result["messages"]:
        message.pretty_print()
