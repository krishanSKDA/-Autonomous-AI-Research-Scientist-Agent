from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    # Input
    query: str                          # original research question

    # Planner output
    sub_questions: List[str]            # decomposed questions
    search_keywords: List[str]          # keywords for search

    # Search agent output
    paper_urls: List[str]               # found paper URLs
    paper_metadata: List[dict]          # title, authors, abstract, year

    # Reader agent output
    extracted_chunks: List[str]         # relevant text chunks
    rag_answers: List[str]              # answers to sub_questions

    # Critic agent output
    relevance_scores: List[float]       # 0-1 score per paper
    flagged_gaps: List[str]             # identified research gaps

    # Synthesizer output
    synthesis: str                      # merged findings
    knowledge_graph: dict               # key concepts + relationships

    # Writer output
    draft_report: str                   # first draft
    final_report: str                   # after self-reflection pass

    # Control
    iteration: int                      # reflection loop counter
    messages: Annotated[list, add_messages]