from pydantic import BaseModel
from typing import Literal, Optional, List, Tuple

class Citation(BaseModel):
    title: str
    url: str

class RetrievalResult(BaseModel):
    text: str
    metadata: dict
    similarity: float
    source_store: Literal["vector", "graph"]

class PipelineResponse(BaseModel):
    answer: str
    source: Literal["local_kb", "web", "none"]
    citations: List[Citation] = []
    needs_permission: bool = False
    raw_query: str

class ExtractedEntities(BaseModel):
    entities: List[str]
    relationships: List[Tuple[str, str, str]]
