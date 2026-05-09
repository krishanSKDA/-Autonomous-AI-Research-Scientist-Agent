import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import ResearchState
from src.tools.arxiv_tool import search_arxiv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research search agent.
Given search keywords, select the BEST 5 keywords to use for academic search.
Respond ONLY as a JSON array of strings.
Example: ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]"""),
    ("human", "Keywords: {keywords}")
])

def search_agent(state: ResearchState) -> ResearchState:
    """Searches ArXiv using keywords and collects paper metadata."""

    # Step 1: LLM picks best keywords
    chain = prompt | llm
    response = chain.invoke({
        "keywords": ", ".join(state["search_keywords"])
    })

    import json
    top_keywords = json.loads(response.content.strip())
    search_query = " ".join(top_keywords[:3])  # combine top 3

    print(f"🔍 Searching ArXiv for: {search_query}")

    # Step 2: Search ArXiv
    papers = search_arxiv.invoke({
        "query": search_query,
        "max_results": 10
    })

    # Step 3: Extract URLs and metadata
    paper_urls = [p["url"] for p in papers]
    paper_metadata = papers

    print(f"📄 Found {len(papers)} papers")
    for p in papers:
        print(f"  - {p['title'][:60]}...")

    return {
        **state,
        "paper_urls": paper_urls,
        "paper_metadata": paper_metadata
    }