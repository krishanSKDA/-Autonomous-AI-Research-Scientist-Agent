import arxiv
from langchain.tools import tool

@tool
def search_arxiv(query: str, max_results: int =10) -> list[dict]:
    """Search ArXiv for academic papers on a topic."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "abstract": paper.summary,
            "url": paper.pdf_url,
            "authors": [a.name for a in paper.authors],
            "published": str(paper.published.date())
        })
    return results