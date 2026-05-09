import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Persistent local ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # lightweight, fast, free
)

def get_collection(name: str = "research_papers"):
    """Get or create a ChromaDB collection."""
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn
    )

def store_chunks(chunks: list[str], paper_title: str, collection_name: str = "research_papers"):
    """Store text chunks into ChromaDB."""
    collection = get_collection(collection_name)
    ids = [f"{paper_title[:30]}_{i}" for i, _ in enumerate(chunks)]
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[{"source": paper_title}] * len(chunks)
    )
    print(f"✅ Stored {len(chunks)} chunks from: {paper_title[:40]}")

def query_collection(question: str, n_results: int = 5, collection_name: str = "research_papers") -> list[str]:
    """Retrieve most relevant chunks for a question."""
    collection = get_collection(collection_name)
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]  # top matching chunks