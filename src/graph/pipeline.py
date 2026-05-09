from langgraph.graph import StateGraph, END
from src.graph.state import ResearchState
from src.agents.planner import planner_agent
from src.agents.search import search_agent
from src.agents.reader import reader_agent
from src.agents.critic import critic_agent
from src.agents.synthesizer import synthesizer_agent
from src.agents.writer import writer_agent

builder = StateGraph(ResearchState)

# Add all nodes
builder.add_node("planner",     planner_agent)
builder.add_node("search",      search_agent)
builder.add_node("reader",      reader_agent)
builder.add_node("critic",      critic_agent)
builder.add_node("synthesizer", synthesizer_agent)
builder.add_node("writer",      writer_agent)

# Wire the pipeline
builder.set_entry_point("planner")
builder.add_edge("planner",     "search")
builder.add_edge("search",      "reader")
builder.add_edge("reader",      "critic")
builder.add_edge("critic",      "synthesizer")
builder.add_edge("synthesizer", "writer")
builder.add_edge("writer",      END)

graph = builder.compile()