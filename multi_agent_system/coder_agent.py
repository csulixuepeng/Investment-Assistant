"""LLM node for reasoning and tool selection."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .config import LLM_CONFIG, CODER_ASSISTANT_PROMPT_TEMPLATE
from .state import State
from .utils import show_graph
from .coder_tools import get_coder_tools



def create_coder_agent():
    # Initialize the LLM
    llm = ChatOpenAI(**LLM_CONFIG)
    
    coder_tools = get_coder_tools()
    
    # Bind tools to LLM
    llm_with_coder_tools = llm.bind_tools(coder_tools)
    
    # Create ToolNode for executing tools
    coder_tool_node = ToolNode(coder_tools)
    
    def coder_assistant(state: State, config: RunnableConfig):
        """LLM node for reasoning and tool selection."""
        memory = "None" 
        if "loaded_memory" in state: 
            memory = state["loaded_memory"]

        coder_assistant_prompt = CODER_ASSISTANT_PROMPT_TEMPLATE
        
        response = llm_with_coder_tools.invoke(
            [SystemMessage(coder_assistant_prompt)] + state["messages"]
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
    coder_workflow = StateGraph(State)
    
    # Add nodes
    coder_workflow.add_node("coder_assistant", coder_assistant)
    coder_workflow.add_node("coder_tool_node", coder_tool_node)
    
    # Add edges
    coder_workflow.add_edge(START, "coder_assistant")
    coder_workflow.add_conditional_edges(
        "coder_assistant",
        should_continue,
        {
            "continue": "coder_tool_node",
            "end": END,
        },
    )
    coder_workflow.add_edge("coder_tool_node", "coder_assistant")
    
    # Initialize memory components
    checkpointer = MemorySaver()
    in_memory_store = InMemoryStore()
    
    # Compile the graph
    coder_subagent = coder_workflow.compile(
        name="coder_subagent",
        checkpointer=checkpointer,
        store=in_memory_store
    )
    
    return coder_subagent


if __name__ == "__main__":
    import uuid
    from langchain_core.messages import HumanMessage
    
    # Create the agent
    coder_agent = create_coder_agent()
    
    # Test with a sample query
    thread_id = uuid.uuid4()
    question = "What's the square root of 42?"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Thread ID: {thread_id}")
    print(f"Question: {question}\n")
    
    # Invoke the agent
    result = coder_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    
    # Print results
    print("Response:")
    for message in result["messages"]:
        message.pretty_print()
