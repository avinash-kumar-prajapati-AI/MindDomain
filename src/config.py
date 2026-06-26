import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class LLMConfig:
    def __init__(self, data: dict):
        self.model = data.get("model", "llama-3.3-70b-versatile")
        self.temperature = float(data.get("temperature", 0.2))
        self.max_tokens = int(data.get("max_tokens", 1024))

class EmbeddingConfig:
    def __init__(self, data: dict):
        self.model = data.get("model", "BAAI/bge-small-en-v1.5")

class RetrievalConfig:
    def __init__(self, data: dict):
        self.vector_threshold = float(data.get("vector_threshold", 0.55))
        self.top_k_vector = int(data.get("top_k_vector", 4))
        self.top_k_graph = int(data.get("top_k_graph", 3))

class WebSearchConfig:
    def __init__(self, data: dict):
        self.provider = data.get("provider", "tavily")
        self.max_results = int(data.get("max_results", 5))

class StorageConfig:
    def __init__(self, data: dict):
        # Resolve to absolute paths relative to project root
        base_dir = Path(__file__).resolve().parent.parent
        self.chroma_path = str((base_dir / data.get("chroma_path", "./data/chroma_db")).resolve())
        self.kuzu_path = str((base_dir / data.get("kuzu_path", "./data/kuzu_db")).resolve())
        self.sqlite_path = str((base_dir / data.get("sqlite_path", "./logs/interactions.db")).resolve())

class DedupConfig:
    def __init__(self, data: dict):
        self.similarity_threshold = float(data.get("similarity_threshold", 0.95))

class Config:
    def __init__(self, yaml_path: str = None):
        if yaml_path is None:
            yaml_path = Path(__file__).resolve().parent.parent / "config.yaml"
        
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
            
        self.llm = LLMConfig(data.get("llm", {}))
        self.embedding = EmbeddingConfig(data.get("embedding", {}))
        self.retrieval = RetrievalConfig(data.get("retrieval", {}))
        self.web_search = WebSearchConfig(data.get("web_search", {}))
        self.storage = StorageConfig(data.get("storage", {}))
        self.dedup = DedupConfig(data.get("dedup", {}))

config = Config()
