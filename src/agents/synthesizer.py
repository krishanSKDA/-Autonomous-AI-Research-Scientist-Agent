import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import ResearchState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research synthesizer agent.
Merge all findings into a coherent synthesis and build a knowledge graph.

Respond ONLY in this JSON format:
{{
  "synthesis": "Detailed merged findings as flowing paragraphs...",
  "knowledge_graph": {{
    "core_concept": "main topic",
    "key_themes": ["theme1", "theme2", "theme3"],
    "methods": ["method1", "method2"],
    "datasets": ["dataset1", "dataset2"],
    "relationships": [
      {{"from": "concept1", "to": "concept2", "relation": "enables"}},
      {{"from": "concept2", "to": "concept3", "relation": "improves"}}
    ]
  }}
}}"""),
    ("human", """Research query: {query}

Sub-questions and answers:
{qa_pairs}

Research gaps identified:
{gaps}

Highly relevant papers (score > 0.6):
{relevant_papers}

Synthesize all findings into a coherent summary.""")
])

def synthesizer_agent(state: ResearchState) -> ResearchState:
    """Merges all findings and builds a knowledge graph."""

    print("\n🧠 Synthesizer agent merging findings...")

    # filter high relevance papers only
    relevant_papers = [
        meta["title"] for meta, score in zip(
            state["paper_metadata"][:5],
            state["relevance_scores"]
        )
        if score >= 0.6
    ]

    qa_pairs = "\n\n".join([
        f"Q: {q}\nA: {a}"
        for q, a in zip(state["sub_questions"], state["rag_answers"])
    ])

    chain = prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "qa_pairs": qa_pairs,
        "gaps": "\n".join(f"- {g}" for g in state["flagged_gaps"]),
        "relevant_papers": "\n".join(f"- {p}" for p in relevant_papers)
    })

    parsed = json.loads(response.content.strip())

    print("  ✅ Synthesis complete")
    print(f"  ✅ Knowledge graph: {len(parsed['knowledge_graph']['key_themes'])} themes")

    return {
        **state,
        "synthesis": parsed["synthesis"],
        "knowledge_graph": parsed["knowledge_graph"]
    }