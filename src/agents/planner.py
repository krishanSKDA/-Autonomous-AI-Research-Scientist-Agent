from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import ResearchState
import os, json
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research planning agent.
Given a research query, decompose it into:
1. 3-5 specific sub-questions
2. 5-8 search keywords for academic databases

Respond ONLY in this JSON format:
{{
  "sub_questions": ["...", "..."],
  "search_keywords": ["...", "..."]
}}"""),
    ("human", "Research query: {query}")
])

def planner_agent(state: ResearchState) -> ResearchState:
    """Decomposes the research query into sub-questions and keywords."""
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})

    # parse JSON response
    content = response.content.strip()
    parsed = json.loads(content)

    return {
        **state,
        "sub_questions": parsed["sub_questions"],
        "search_keywords": parsed["search_keywords"],
        "iteration": 0
    }