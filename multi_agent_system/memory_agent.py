from langgraph.store.base import BaseStore # Base class for defining custom stores for LangGraph
from langchain_core.runnables import RunnableConfig # Configuration class for runnable nodes in LangGraph
from .state import State # Importing the State schema defined for our multi-agent system
from .config import LLM_CONFIG, CREATE_MEMORY_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from .user_file import UserProfile


def create_memory(state: State, config: RunnableConfig, store: BaseStore):
     # Initialize the LLM
    llm = ChatOpenAI(**LLM_CONFIG)
    
    # Get the customer ID from the state, with fallback to config
    user_id = state.get("customer_id")
    
    # If customer_id is not in state, try to get it from config
    if not user_id:
        if isinstance(config, dict):
            user_id = config.get("configurable", {}).get("user_id", "unknown")
        elif hasattr(config, 'configurable'):
            user_id = config.configurable.get("user_id", "unknown") if isinstance(config.configurable, dict) else "unknown"
        else:
            user_id = "unknown"
    
    user_id = str(user_id) # Convert to string
    namespace = ("memory_profile", user_id) # Define the namespace for this user's memory profile.
    
    # Retrieve the existing memory profile for this user from the long-term store.
    existing_memory = store.get(namespace, "user_memory")
    
    formatted_memory = "" # Initialize formatted memory for the prompt.
    if existing_memory and existing_memory.value:
        existing_memory_dict = existing_memory.value # Get the dictionary containing the UserProfile instance.
        # Format existing invest preferences into a string for the prompt.
        formatted_memory = (
            f"Invest Preferences: {', '.join(existing_memory_dict.get('memory').invest_preferences or [])}" # Access the UserProfile object via 'memory' key
        )

    # Format the conversation history as a string
    conversation_str = ""
    for msg in state["messages"]:
        if hasattr(msg, 'content'):
            conversation_str += f"{msg.__class__.__name__}: {msg.content}\n"
        else:
            conversation_str += f"{str(msg)}\n"
    
    # Create a SystemMessage with the formatted prompt, injecting the full conversation history
    # and the existing memory profile.
    formatted_system_message = SystemMessage(
        content=CREATE_MEMORY_PROMPT.format(conversation=conversation_str, memory_profile=formatted_memory)
    )
    
    # Create a modified message that includes JSON instruction for OpenAI API
    modified_message = HumanMessage(
        content="请分析对话内容，并以包含更新后内存配置文件的 JSON 对象的形式回复。 "
                "请务必将 customer_id（如果未提及，则使用提供的 user_id）和 invest_preferences 作为列表包含在内。"
    )
    
    # Invoke the LLM with structured output (`UserProfile`) to analyze the conversation
    # and update the memory profile based on new information.
    try:
        updated_memory = llm.with_structured_output(UserProfile).invoke([formatted_system_message, modified_message])
    except Exception as e:
        # Fallback: create a simple UserProfile if structured output fails
        print(f"Warning: Structured output failed, using fallback. Error: {e}")
        updated_memory = UserProfile(customer_id=user_id, invest_preferences=[])
    
    key = "user_memory" # Define the key for storing this specific memory object.
    
    # Store the updated memory profile back into the `InMemoryStore`.
    # We wrap `updated_memory` in a dictionary under the key 'memory' for consistency in access.
    store.put(namespace, key, {"memory": updated_memory})


