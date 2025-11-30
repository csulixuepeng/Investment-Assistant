from langgraph.store.base import BaseStore # Base class for defining custom stores for LangGraph
from langchain_core.runnables import RunnableConfig # Configuration class for runnable nodes in LangGraph
from .state import State # Importing the State schema defined for our multi-agent system

# Helper function to format user memory into a readable string.
def format_user_memory(user_data):
    """Formats invest preferences from users, if available."""
    profile = user_data['memory'] # Access the 'memory' key from the stored dictionary
    result = "" # Initialize an empty string for the formatted result
    
    if hasattr(profile, 'invest_preferences') and profile.invest_preferences:
        result += f"Invest Preferences: {', '.join(profile.invest_preferences)}"
    
    return result.strip() # Return the formatted string, removing any leading/trailing whitespace.

# Define the `load_memory` node function.
# This node loads a user's long-term memory into the current state.
def load_memory(state: State, config: RunnableConfig, store: BaseStore):
    """Loads invest preferences from users, if available."""
    
    # Get the current customer ID from the state, with a default value if not provided
    user_id = state.get("customer_id")
    
    # If customer_id is not in state, try to get it from config
    if not user_id:
        if isinstance(config, dict):
            user_id = config.get("configurable", {}).get("user_id", "unknown")
        elif hasattr(config, 'configurable'):
            user_id = config.configurable.get("user_id", "unknown") if isinstance(config.configurable, dict) else "unknown"
        else:
            user_id = "unknown"
    
    namespace = ("memory_profile", user_id) # Define a namespace for storing user-specific memory.
                                          # This creates a unique key for each user's profile.
    
    # Attempt to retrieve existing memory for this user from the `InMemoryStore`.
    existing_memory = store.get(namespace, "user_memory")
    
    formatted_memory = "" # Initialize formatted memory as empty.
    
    # If memory exists and has a value, format it using our helper function.
    if existing_memory and existing_memory.value:
        formatted_memory = format_user_memory(existing_memory.value)

    # Update the `loaded_memory` field in the state with the retrieved and formatted memory.
    return {"loaded_memory" : formatted_memory, "existing_memory": existing_memory}