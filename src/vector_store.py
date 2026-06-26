import uuid
import chromadb
from typing import List
from src.config import config
from src.models import RetrievalResult

_client = None
_collection = None

def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.storage.chroma_path)
    return _client

def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        # Create or get collection using cosine similarity metric
        _collection = client.get_or_create_collection(
            name="graphrag_chunks",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def add(text: str, embedding: List[float], metadata: dict) -> str:
    """
    Inserts a chunk text and its vector embedding into Chroma DB.
    Returns the generated unique chunk ID.
    """
    collection = get_collection()
    chunk_id = str(uuid.uuid4())
    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )
    return chunk_id

def search(embedding: List[float], top_k: int) -> List[RetrievalResult]:
    """
    Searches Chroma DB for similar passages and maps to RetrievalResult.
    Cosine similarity is derived from Chroma's cosine distance (1.0 - distance).
    """
    collection = get_collection()
    
    # Avoid querying if the collection is empty
    if collection.count() == 0:
        return []
        
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count())
    )
    
    retrieval_results = []
    if not results or not results["ids"] or not results["ids"][0]:
        return retrieval_results
        
    ids = results["ids"][0]
    distances = results["distances"][0] if results["distances"] is not None else [0.0] * len(ids)
    documents = results["documents"][0] if results["documents"] is not None else [""] * len(ids)
    metadatas = results["metadatas"][0] if results["metadatas"] is not None else [{}] * len(ids)
    
    for i in range(len(ids)):
        # Chroma cosine distance is 1.0 - cosine_similarity
        distance = distances[i]
        similarity = 1.0 - distance
        
        meta = metadatas[i] or {}
        meta["chunk_id"] = ids[i]
        
        retrieval_results.append(RetrievalResult(
            text=documents[i],
            metadata=meta,
            similarity=similarity,
            source_store="vector"
        ))
        
    return retrieval_results

def is_near_duplicate(embedding: List[float], threshold: float) -> bool:
    """
    Checks if a chunk with similar embedding already exists above similarity threshold.
    """
    collection = get_collection()
    
    if collection.count() == 0:
        return False
        
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    
    if not results or not results["distances"] or not results["distances"][0]:
        return False
        
    closest_distance = results["distances"][0][0]
    closest_similarity = 1.0 - closest_distance
    return closest_similarity >= threshold
