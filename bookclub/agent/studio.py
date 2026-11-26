"""Graph exposed to the LangGraph CLI (`langgraph dev`) and LangGraph Studio, declared in langgraph.json.

The LangGraph server manages threads and checkpoints itself, so the graph is compiled without a checkpointer.
"""

from bookclub.agent.runtime import build_agent

graph = build_agent()
