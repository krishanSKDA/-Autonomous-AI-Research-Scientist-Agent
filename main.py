from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid, asyncio
from src.graph.pipeline import graph

app = FastAPI(title="Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory job store
jobs: dict = {}

class ResearchRequest(BaseModel):
    query: str

class JobStatus(BaseModel):
    job_id: str
    status: str        # pending | running | done | failed
    result: dict = {}

# ── run pipeline in background ──
def run_pipeline(job_id: str, query: str):
    jobs[job_id]["status"] = "running"
    try:
        result = graph.invoke({
            "query": query,
            "sub_questions": [],
            "search_keywords": [],
            "paper_urls": [],
            "paper_metadata": [],
            "extracted_chunks": [],
            "rag_answers": [],
            "relevance_scores": [],
            "flagged_gaps": [],
            "synthesis": "",
            "knowledge_graph": {},
            "draft_report": "",
            "final_report": "",
            "iteration": 0,
            "messages": []
        })

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "query":            result["query"],
            "sub_questions":    result["sub_questions"],
            "papers":           result["paper_metadata"],
            "relevance_scores": result["relevance_scores"],
            "flagged_gaps":     result["flagged_gaps"],
            "knowledge_graph":  result["knowledge_graph"],
            "final_report":     result["final_report"],
            "iteration":        result["iteration"]
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = {"error": str(e)}

# ── endpoints ──
@app.post("/research", response_model=JobStatus)
async def start_research(req: ResearchRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": {}}
    bg.add_task(run_pipeline, job_id, req.query)
    return JobStatus(job_id=job_id, status="pending")

@app.get("/research/{job_id}", response_model=JobStatus)
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JobStatus(job_id=job_id, status="not_found")
    return JobStatus(job_id=job_id, **job)

@app.get("/health")
async def health():
    return {"status": "ok"}