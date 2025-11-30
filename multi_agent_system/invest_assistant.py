from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from multi_agent_system.state import State
from multi_agent_system.utils import show_graph
from multi_agent_system.memory_tool import load_memory
from multi_agent_system.supervisor import create_supervisor
from multi_agent_system.memory_agent import create_memory
from langgraph.checkpoint.memory import MemorySaver # For short-term memory (thread-level state persistence)
from langgraph.store.memory import InMemoryStore # For long-term memory (storing user preferences)
from multi_agent_system.config import LLM_CONFIG


def create_invest_assistant():
    """Creates a multi-agent investment assistant with memory management."""
     # Initialize the LLM
    llm = ChatOpenAI(**LLM_CONFIG)

    in_memory_store = InMemoryStore()

    checkpointer = MemorySaver()

    invest_assistan_agent = StateGraph(State)

    invest_assistan_agent.add_node("load_memory", load_memory) 
    invest_assistan_agent.add_node("supervisor", create_supervisor)  # Call the function to get the compiled graph
    invest_assistan_agent.add_node("create_memory", create_memory)

    invest_assistan_agent.add_edge(START, "load_memory")
    invest_assistan_agent.add_edge("load_memory", "supervisor")
    invest_assistan_agent.add_edge("supervisor", "create_memory")
    invest_assistan_agent.add_edge("create_memory", END)

    # Compile the entire, sophisticated graph.
    invest_assistant_workflow = invest_assistan_agent.compile(name="multi_agent_verify", checkpointer=checkpointer, store=in_memory_store)
    
    return invest_assistant_workflow

if __name__ == "__main__":
    """Test the invest assistant independently."""
    import uuid
    from langchain_core.messages import HumanMessage
    
    # Create the invest assistant
    invest_assistant = create_invest_assistant()

    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id, "user_id" : "1"}}

    question = "对于一位30岁、风险承受能力中等的年轻人来说，有哪些好的投资选择？请提供一些具体的投资产品建议，并解释为什么这些选择适合该投资者。"

    result = invest_assistant.invoke({"messages": [HumanMessage(content=question)]}, config=config)

    # Print results
    print("Response:")
    for message in result["messages"]:
        message.pretty_print() 