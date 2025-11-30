"""Search tools for web search and information retrieval."""
from typing import Annotated

from langchain_core.tools import tool
from tavily import TavilyClient

# Initialize Tavily client for web search
tavily_client = TavilyClient(api_key="")


@tool
def tavily_search(
    query: Annotated[str, "The search query to execute."],
) -> str:
    """Search the web for information using Tavily search engine."""
    try:
        result = tavily_client.search(query=query, max_results=5)
        return str(result)
    except Exception as e:
        return f"Search failed: {str(e)}"
    

@tool
def tavily_get_page_content(
    url: Annotated[str, "The URL of the web page to retrieve content from."],
) -> str:
    """Retrieve the content of a web page using Tavily."""
    try:
        content = tavily_client.extract(url=url)
        return content
    except Exception as e:
        return f"Failed to retrieve page content: {str(e)}" 


# Aggregate all searcher-related tools into a list
searcher_tools = [
    tavily_search,
    tavily_get_page_content
]


def get_searcher_tools():
    """Get the list of searcher tools."""
    return searcher_tools
