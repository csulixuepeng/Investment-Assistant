from pydantic import BaseModel, Field
from typing import List, Annotated
from typing_extensions import TypedDict
from .state import State
import os


def show_graph(graph, xray=False):
    """
    Display a LangGraph mermaid diagram with fallback rendering.
    
    This function attempts to render a LangGraph as a visual diagram using Mermaid.
    It includes error handling to fall back to an alternative renderer if the default fails.
    
    Args:
        graph: The LangGraph object that has a get_graph() method for visualization
        xray (bool): Whether to show internal graph details in xray mode
        
    Returns:
        Image: An IPython Image object containing the rendered graph diagram
    """
    from IPython.display import Image
    
    try:
        # Try the default mermaid renderer first (uses mermaid.ink service)
        # This is the fastest option but may fail due to network issues or service unavailability
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        # If the default renderer fails, fall back to pyppeteer
        # pyppeteer uses a local headless Chrome instance to render the diagram
        print(f"Default renderer failed ({e}), falling back to pyppeteer...")
        
        # Apply nest_asyncio to handle async operations in Jupyter environments
        # This is necessary because pyppeteer uses async operations
        import nest_asyncio
        nest_asyncio.apply()
        
        # Import the MermaidDrawMethod enum for specifying the draw method
        from langchain_core.runnables.graph import MermaidDrawMethod
        
        # Use pyppeteer as the drawing method (local rendering)
        return Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))




