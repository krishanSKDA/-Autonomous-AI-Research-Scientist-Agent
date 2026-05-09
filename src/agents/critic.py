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
    ("system", """You are a research critic agent.
Evaluate the research findings and identify gaps.

Respond ONLY in this JSON format:
{{
  "relevance_scores": [0.9, 0.7, 0.8, 0.6, 0.5],
  "flagged_gaps": [
    "Gap 1 description",
    "Gap 2 description",
    "Gap 3 description"
  ]
}}

Relevance scores must be between 0.0 and 1.0.
Provide exactly 3-5 research gaps."""),
    ("human", """Original query: {query}

Sub-questions asked:
{sub_questions}

RAG answers received:
{rag_answers}

Paper titles found:
{paper_titles}

Evaluate relevance of each paper and identify research gaps.""")
])

def critic_agent(state: ResearchState) -> ResearchState:
    """Scores paper relevance and flags research gaps."""

    print("\n🔍 Critic agent evaluating findings...")

    paper_titles = [p["title"] for p in state["paper_metadata"][:5]]

    chain = prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "sub_questions": "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(state["sub_questions"])
        ),
        "rag_answers": "\n".join(
            f"Q{i+1}: {a[:200]}" for i, a in enumerate(state["rag_answers"])
        ),
        "paper_titles": "\n".join(
            f"- {t}" for t in paper_titles
        )
    })

    parsed = json.loads(response.content.strip())

    scores = parsed["relevance_scores"]
    gaps = parsed["flagged_gaps"]

    print(f"  ✅ Scored {len(scores)} papers")
    print(f"  ✅ Found {len(gaps)} research gaps")
    for gap in gaps:
        print(f"     ⚠️  {gap[:70]}")

    return {
        **state,
        "relevance_scores": scores,
        "flagged_gaps": gaps
    }