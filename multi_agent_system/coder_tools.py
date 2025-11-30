from typing import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# This executes code locally, which can be unsafe
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Execute Python code and return the result."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str



coder_tools = [
    python_repl_tool
]


def get_coder_tools():
    return coder_tools