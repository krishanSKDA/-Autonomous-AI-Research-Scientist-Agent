import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import ResearchState
from src.tools.pdf_tool import read_pdf_from_url
from src.memory.vector_store import store_chunks, query_collection

load_dotenv()

# Gemini for Reader — long context handles big PDFs well
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research reader agent.
Given relevant paper excerpts, answer the research question concisely.
Focus on key findings, methodologies, and results.
Be specific and cite evidence from the text."""),
    ("human", """Research question: {question}

Relevant excerpts:
{context}

Provide a focused answer based only on these excerpts.""")
])

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def reader_agent(state: ResearchState) -> ResearchState:
    """Reads PDFs, stores in ChromaDB, answers sub-questions via RAG."""

    extracted_chunks = []
    rag_answers = []

    # Step 1: Read each paper and store in ChromaDB
    for i, (url, meta) in enumerate(zip(
        state["paper_urls"][:5],        # limit to top 5 papers
        state["paper_metadata"][:5]
    )):
        print(f"\n📖 Reading paper {i+1}/5: {meta['title'][:50]}...")
        try:
            raw_text = read_pdf_from_url.invoke({"url": url})
            chunks = chunk_text(raw_text)
            store_chunks(chunks, paper_title=meta["title"])
            extracted_chunks.extend(chunks[:3])  # keep sample chunks in state
            print(f"   ✅ {len(chunks)} chunks stored")
        except Exception as e:
            print(f"   ⚠️  Skipped: {e}")
            continue

    # Step 2: Answer each sub-question via RAG
    print("\n🤔 Answering sub-questions via RAG...")
    chain = prompt | llm

    for question in state["sub_questions"]:
        print(f"  Q: {question[:60]}...")

        # Retrieve relevant chunks from ChromaDB
        relevant_chunks = query_collection(question, n_results=5)
        context = "\n\n---\n\n".join(relevant_chunks)

        # LLM answers based on retrieved context
        response = chain.invoke({
            "question": question,
            "context": context
        })

        rag_answers.append(response.content)
        print(f"  ✅ Answered")

    return {
        **state,
        "extracted_chunks": extracted_chunks,
        "rag_answers": rag_answers
    }