"""
State definition for the multi-agent system.
Defines the shared state schema used across all agents.
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps


class State(TypedDict):
    """Represents the state of our LangGraph agent."""
    # customer_id: Stores the unique identifier for the current customer.
    customer_id: str
    
    # messages: A list of messages that form the conversation history.
    # Annotated with `add_messages` to ensure new messages are appended rather than overwritten.
    messages: Annotated[list[AnyMessage], add_messages]
    
    # loaded_memory: Stores information loaded from the long-term memory store, 
    # typically user preferences or historical context.
    loaded_memory: str
    
    # remaining_steps: Used by LangGraph to track the number of allowed steps 
    # to prevent infinite loops in cyclic graphs.
    remaining_steps: RemainingSteps
