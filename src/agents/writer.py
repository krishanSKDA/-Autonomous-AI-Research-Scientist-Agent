import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import ResearchState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# First draft prompt
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research writer agent.
Write a structured research report in IEEE-style format with these sections:

1. Abstract (150 words)
2. Introduction
3. Key Findings
4. Research Gaps & Future Directions
5. Conclusion

Use clear academic language. Reference paper titles where relevant."""),
    ("human", """Query: {query}

Synthesis: {synthesis}

Knowledge Graph themes: {themes}

Research Gaps: {gaps}

Write the full structured report.""")
])

# Self-reflection prompt
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research quality reviewer.
Review this draft report and improve it by:
- Fixing any contradictions or vague claims
- Strengthening the gap analysis section
- Making the conclusion more actionable
- Ensuring academic tone throughout

Return ONLY the improved final report, no commentary."""),
    ("human", """Original query: {query}

Draft report to improve:
{draft}

Return the improved final version.""")
])

def writer_agent(state: ResearchState) -> ResearchState:
    """Writes draft, reflects on it, produces final report."""

    print("\n✍️  Writer agent drafting report...")

    chain = draft_prompt | llm
    draft_response = chain.invoke({
        "query": state["query"],
        "synthesis": state["synthesis"],
        "themes": ", ".join(state["knowledge_graph"].get("key_themes", [])),
        "gaps": "\n".join(f"- {g}" for g in state["flagged_gaps"])
    })

    draft = draft_response.content
    print("  ✅ Draft complete")
    print(f"  📝 Draft length: {len(draft.split())} words")

    # Self-reflection pass
    print("  🔄 Running self-reflection pass...")
    reflect_chain = reflection_prompt | llm
    final_response = reflect_chain.invoke({
        "query": state["query"],
        "draft": draft
    })

    final = final_response.content
    print(f"  ✅ Final report: {len(final.split())} words")

    return {
        **state,
        "draft_report": draft,
        "final_report": final,
        "iteration": state["iteration"] + 1
    }