"""
Multi-Agent System Package

A modular, scalable multi-agent system using LangGraph and LangChain.
Provides intelligent investment analysis through specialized agents.
"""

__version__ = "1.0.0"
__author__ = "Investment Assistant Contributors"

from .invest_assistant import create_invest_assistant
from .supervisor import create_supervisor
from .searcher_agent import create_searcher_agent
from .summary_agent import create_summary_agent
from .coder_agent import create_coder_agent
from .memory_agent import create_memory

__all__ = [
    "create_invest_assistant",
    "create_supervisor",
    "create_searcher_agent",
    "create_summary_agent",
    "create_coder_agent",
    "create_memory",
]
