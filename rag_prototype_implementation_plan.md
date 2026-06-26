# Self-Improving Local GraphRAG Engine — Prototype Plan v2

Target hardware: 8GB VRAM, 16GB RAM, Windows + Miniconda.
LLM backend: llama-server (already installed, runs as local HTTP server). Models: Qwen2.5-3B or Gemma-3-4B (GGUF Q4_K_M).
Retrieval: Hybrid — Chroma vector store + Kuzu graph DB.
Interface: Terminal REPL.

---

## Why graph DB alongside the vector store?

Pure vector similarity answers "what chunk is closest to this question" but breaks down on:
- Entity relationships ("what games are in the Fable series and who made each one?")
- Multi-hop facts ("who was the director of the studio that made Fable 5?")
- Topic clustering ("show me everything I have stored about Microsoft game studios")

The graph stores entities (Game, Studio, Person, Topic) and named relationships (DEVELOPED_BY, SEQUEL_OF, PUBLISHED_BY). When a question arrives, the pipeline first traverses the graph to pull a structured context subgraph, then searches the vector store for raw chunk similarity, then merges both contexts before calling the LLM. This makes complex retrieval far more reliable than vector-only approaches at minimal extra cost — Kuzu is an embedded graph DB (no server) and adds under 50MB to your install.

---

## 1. Tech stack

| Role | Library / Tool | Notes |
|---|---|---|
| LLM server | `llama-server.exe` (already installed) | Start once before running the engine |
| LLM client | `httpx` | Calls llama-server's OpenAI-compatible HTTP API |
| Model | Qwen2.5-3B-Instruct Q4_K_M or Gemma-3-4B-IT Q4_K_M GGUF | Both fit in 8GB VRAM with room to spare |
| Embeddings | `sentence-transformers` `BAAI/bge-small-en-v1.5` | CPU only, fast |
| Vector store | `chromadb` | File-based, no server |
| Graph DB | `kuzu` | Embedded, no server, persists to disk |
| Web search | `tavily-python` (free tier) | Fallback: `duckduckgo-search` |
| Entity extraction | Prompt the local LLM via llama-server | No extra NLP libraries needed |
| Config | `pyyaml` + `python-dotenv` | |
| Data models | `pydantic` | |
| Log | built-in `sqlite3` | |
| Interface | Plain terminal REPL (`prompt_toolkit` for history/autocomplete) | |

---

## 2. Miniconda environment setup

**One-time: start llama-server before running the engine**

Since you already have llama.cpp built, just run this in a separate terminal and leave it running:

```bash
llama-server.exe -m ./models/gemma-4-E4B-it-Q8_0.gguf -ngl 99 --port 8080 --ctx-size 4096
```

Flags explained:
- `-ngl 99` — offload all layers to GPU (99 is high enough to cover any model size)
- `--ctx-size 4096` — context window; reduce to 2048 if you get VRAM OOM errors
- `--port 8080` — where Python will talk to it

Verify it's up: open `http://localhost:8080/health` in a browser — it should return `{"status":"ok"}`.

**Python environment (no llama-cpp-python needed)**

```bash
conda create -n ragengine python=3.11
conda activate ragengine

pip install httpx chromadb kuzu sentence-transformers tavily-python duckduckgo-search pydantic python-dotenv pyyaml prompt_toolkit rich
```

That's it — no CUDA wheels, no C++ compilation, no version conflicts. The heavy lifting stays in the llama-server process you already know works.

**Model download:**
Pick one. Both work on 8GB VRAM:
- `Qwen2.5-3B-Instruct-Q4_K_M.gguf` — from Hugging Face `Qwen/Qwen2.5-3B-Instruct-GGUF`
- `gemma-3-4b-it-Q4_K_M.gguf` — from Hugging Face `google/gemma-3-4b-it-GGUF`

Put the downloaded `.gguf` file in the `models/` directory.

---

## 3. Project structure

```
rag-engine/
├── .env
├── config.yaml
├── requirements.txt
├── models/
│   └── qwen2.5-3b-instruct-q4_k_m.gguf   ← or gemma
├── data/
│   ├── chroma_db/      ← vector store (auto-created)
│   └── kuzu_db/        ← graph store (auto-created)
├── logs/
│   └── interactions.db ← sqlite log (auto-created)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py           ← pydantic data models
│   ├── embedder.py
│   ├── vector_store.py     ← Chroma wrapper
│   ├── graph_store.py      ← Kuzu wrapper  ← NEW
│   ├── llm_client.py       ← llama-cpp wrapper
│   ├── entity_extractor.py ← uses LLM to pull entities from answers  ← NEW
│   ├── web_search.py
│   ├── prompts.py
│   ├── rag_pipeline.py     ← hybrid retrieval orchestrator
│   └── feedback_store.py
└── main.py                 ← terminal REPL entry point
```

---

## 4. Configuration

`config.yaml`:

```yaml
llm:
  base_url: "http://localhost:8080"   # llama-server address
  temperature: 0.2
  max_tokens: 1024
  stop_tokens: ["</s>", "<|im_end|>", "<end_of_turn>"]  # covers Qwen + Gemma

embedding:
  model: "BAAI/bge-small-en-v1.5"

retrieval:
  vector_threshold: 0.55   # below this, skip straight to "I don't know"
  top_k_vector: 4
  top_k_graph: 3           # max entity-linked chunks to pull from graph traversal

web_search:
  provider: "tavily"        # "tavily" or "duckduckgo"
  max_results: 5

storage:
  chroma_path: "./data/chroma_db"
  kuzu_path: "./data/kuzu_db"
  sqlite_path: "./logs/interactions.db"

dedup:
  similarity_threshold: 0.95
```

`.env`:

```
TAVILY_API_KEY=your_key_here
```

---

## 5. Data models (`src/models.py`)

```python
from pydantic import BaseModel
from typing import Literal, Optional

class Citation(BaseModel):
    title: str
    url: str

class RetrievalResult(BaseModel):
    text: str
    metadata: dict
    similarity: float
    source_store: Literal["vector", "graph"]   # which store surfaced this chunk

class PipelineResponse(BaseModel):
    answer: str
    source: Literal["local_kb", "web", "none"]
    citations: list[Citation] = []
    needs_permission: bool = False
    raw_query: str

class ExtractedEntities(BaseModel):
    entities: list[str]                         # ["Fable 5", "Microsoft", "Xbox Game Studios"]
    relationships: list[tuple[str, str, str]]   # (subject, relation, object)
```

---

## 6. Module specifications

### `llm_client.py`

Talks to `llama-server` over HTTP. No model loading in Python — the server owns that:

```python
import httpx
from src.config import config

# Reuse a single httpx client across all calls (connection pooling, faster)
_client = httpx.Client(base_url=config.llm.base_url, timeout=120.0)

def generate(prompt: str) -> str:
    response = _client.post("/completion", json={
        "prompt": prompt,
        "temperature": config.llm.temperature,
        "n_predict": config.llm.max_tokens,
        "stop": config.llm.stop_tokens,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["content"].strip()

def health_check() -> bool:
    """Call this at startup to fail fast if llama-server isn't running."""
    try:
        r = _client.get("/health")
        return r.json().get("status") == "ok"
    except Exception:
        return False
```

Call `health_check()` at the top of `main.py` before starting the REPL and print a clear error if it returns `False`:

```python
if not llm_client.health_check():
    console.print("[bold red]llama-server is not running.[/]")
    console.print("Start it first: llama-server.exe -m ./models/yourmodel.gguf -ngl 99 --port 8080")
    raise SystemExit(1)
```

This saves a lot of confusing debugging — a missing server shows up as a timeout otherwise.

### `embedder.py`
Load `bge-small-en-v1.5` once at import time via `SentenceTransformer`, run on CPU. One function: `embed(text: str) -> list[float]`. Prefix queries with `"Represent this sentence for searching relevant passages: "` and documents with `"Represent this passage for retrieval: "` — this is required by the BGE model family and improves recall noticeably.

### `vector_store.py`
Persistent Chroma client pointing at `storage.chroma_path`. Collection must be created with `metadata={"hnsw:space": "cosine"}`. Three functions:
- `search(embedding, top_k) -> list[RetrievalResult]`
- `add(text, embedding, metadata) -> str` — returns the generated ID
- `is_near_duplicate(embedding, threshold) -> bool`

### `graph_store.py` — the new component

Kuzu schema (run once at DB creation):

```python
CREATE NODE TABLE Entity (id STRING, name STRING, type STRING, PRIMARY KEY(id))
CREATE NODE TABLE Chunk (id STRING, text STRING, source STRING, timestamp STRING, PRIMARY KEY(id))
CREATE REL TABLE MENTIONED_IN (FROM Entity TO Chunk)
CREATE REL TABLE RELATED_TO (FROM Entity TO Entity, relation STRING)
```

Functions needed:

```python
def ensure_schema() -> None
    # Run CREATE TABLE statements if tables don't exist; call at startup

def add_chunk_with_entities(chunk_id: str, chunk_text: str, 
                             entities: ExtractedEntities, metadata: dict) -> None
    # Insert Chunk node, insert or merge each Entity node,
    # create MENTIONED_IN edges from each entity to the chunk,
    # create RELATED_TO edges from the extracted relationships

def search_by_entity(query_entities: list[str], top_k: int) -> list[RetrievalResult]
    # Find entities in the graph matching query_entities (fuzzy match on name),
    # traverse MENTIONED_IN to get linked Chunks, return top_k unique chunks
    # Also traverse RELATED_TO one hop to get related entity chunks

def get_entity_context(entity_name: str) -> str
    # Return a subgraph summary: what is this entity connected to?
    # Used to enrich the LLM prompt with structured facts
```

The Kuzu query for `search_by_entity`:

```cypher
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE e.name IN $entity_list
RETURN c.id, c.text, c.source
UNION
MATCH (e1:Entity)-[:RELATED_TO]->(e2:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE e1.name IN $entity_list
RETURN c.id, c.text, c.source
LIMIT $top_k
```

### `entity_extractor.py`

Uses the local LLM — no extra NLP library:

```python
ENTITY_EXTRACTION_PROMPT = """Extract named entities and relationships from the text below.
Respond ONLY with valid JSON matching this exact schema, no other text:
{
  "entities": ["entity1", "entity2"],
  "relationships": [["subject", "relation", "object"]]
}

Text:
{text}

JSON:"""

def extract(text: str) -> ExtractedEntities:
    raw = llm_client.generate(ENTITY_EXTRACTION_PROMPT.format(text=text))
    # strip any markdown fences before parsing
    clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(clean)
    return ExtractedEntities(**data)
```

Run entity extraction only on text being *written* to the store (after a thumbs-up), not on every query. This keeps query latency fast.

### `rag_pipeline.py` — hybrid orchestrator

The three stages per query:

**Stage 1 — Entity-aware retrieval**

```python
def build_context(query: str) -> tuple[str, list[RetrievalResult]]:
    query_emb = embedder.embed(query)
    
    # Vector retrieval
    vector_results = vector_store.search(query_emb, top_k=config.retrieval.top_k_vector)
    
    # Graph retrieval: extract entities from the query itself (cheap, just the query string)
    query_entities = entity_extractor.extract(query).entities
    graph_results = graph_store.search_by_entity(query_entities, top_k=config.retrieval.top_k_graph)
    
    # Merge and deduplicate by chunk ID
    all_results = deduplicate(vector_results + graph_results)
    
    context = "\n\n---\n\n".join(r.text for r in all_results)
    return context, all_results
```

**Stage 2 — Confidence check**

```python
def answer(query: str) -> PipelineResponse:
    context, results = build_context(query)
    
    if not results or results[0].similarity < config.retrieval.vector_threshold:
        return PipelineResponse(
            answer="I don't have anything on that. Should I check online? [y/n]",
            source="none", needs_permission=True, raw_query=query
        )
    
    raw = llm_client.generate(
        prompts.ANSWER_FROM_CONTEXT.format(context=context, question=query)
    )
    if raw.strip() == "NOT_FOUND":
        return PipelineResponse(
            answer="My stored knowledge doesn't cover that confidently. Should I check online? [y/n]",
            source="none", needs_permission=True, raw_query=query
        )
    
    return PipelineResponse(answer=raw, source="local_kb", raw_query=query)
```

**Stage 3 — Web search + feedback write-back**

```python
def search_web_and_answer(query: str) -> PipelineResponse:
    results = web_search.search(query, config.web_search.max_results)
    numbered = "\n".join(f"[{i+1}] {r.title}: {r.snippet}" for i, r in enumerate(results))
    raw = llm_client.generate(prompts.WEB_SYNTHESIS.format(results=numbered, question=query))
    citations = [Citation(title=r.title, url=r.url) for r in results]
    return PipelineResponse(answer=raw, source="web", citations=citations, raw_query=query)


def submit_feedback(response: PipelineResponse, rating: str) -> None:
    feedback_store.log(response, rating)
    if rating == "up" and response.source in ("web", "local_kb"):
        chunk_text = f"Q: {response.raw_query}\nA: {response.answer}"
        emb = embedder.embed(chunk_text)
        if not vector_store.is_near_duplicate(emb, config.dedup.similarity_threshold):
            chunk_id = str(uuid4())
            metadata = {"source": response.source, "timestamp": now_iso()}
            
            # Write to vector store
            vector_store.add(chunk_text, emb, metadata)
            
            # Extract entities and write to graph store
            entities = entity_extractor.extract(chunk_text)
            graph_store.add_chunk_with_entities(chunk_id, chunk_text, entities, metadata)
```

---

## 7. Terminal REPL (`main.py`)

Use `prompt_toolkit` for command history (up-arrow recall) and `rich` for colored output. States the REPL tracks:

```
IDLE          → waiting for a new question
AWAITING_PERM → waiting for y/n after "should I check online?"
AWAITING_FB   → waiting for thumbs up (u) / thumbs down (d) after an answer
```

```python
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.markdown import Markdown

console = Console()
session = PromptSession()

def run():
    console.print("[bold cyan]RAG Engine[/] — type a question or 'quit'")
    state = "IDLE"
    last_response = None

    while True:
        if state == "IDLE":
            query = session.prompt("\n[You] ").strip()
            if not query or query.lower() == "quit":
                break
            response = rag_pipeline.answer(query)
            console.print(f"\n[bold green][RAG][/] {response.answer}")
            if response.needs_permission:
                state = "AWAITING_PERM"
            else:
                if response.citations:
                    for i, c in enumerate(response.citations, 1):
                        console.print(f"  [{i}] {c.title} — {c.url}")
                console.print("\n  Was this helpful? [u]p / [d]own")
                state = "AWAITING_FB"
            last_response = response

        elif state == "AWAITING_PERM":
            inp = session.prompt("[y/n] ").strip().lower()
            if inp in ("y", "yes"):
                console.print("[dim]Searching the web...[/]")
                last_response = rag_pipeline.search_web_and_answer(last_response.raw_query)
                console.print(f"\n[bold green][RAG][/] {last_response.answer}")
                if last_response.citations:
                    for i, c in enumerate(last_response.citations, 1):
                        console.print(f"  [{i}] {c.title} — {c.url}")
                console.print("\n  Was this helpful? [u]p / [d]own")
                state = "AWAITING_FB"
            else:
                console.print("[dim]OK, skipping web search.[/]")
                state = "IDLE"

        elif state == "AWAITING_FB":
            inp = session.prompt("[u/d] ").strip().lower()
            if inp in ("u", "up", "d", "down"):
                rating = "up" if inp in ("u", "up") else "down"
                rag_pipeline.submit_feedback(last_response, rating)
                label = "Saved to knowledge base." if rating == "up" else "Logged, not saved."
                console.print(f"[dim]{label}[/]")
            else:
                console.print("[dim]Skipped feedback.[/]")
            state = "IDLE"

if __name__ == "__main__":
    run()
```

---

## 8. Prompts (`src/prompts.py`)

```python
ANSWER_FROM_CONTEXT = """Answer using ONLY the context provided. Do not use outside knowledge.
If the context does not contain enough to answer, reply with exactly: NOT_FOUND

Context:
{context}

Question: {question}
Answer:"""

WEB_SYNTHESIS = """Answer using only the numbered search results. Cite sources inline as [1], [2], etc.
If the results don't answer the question, say so plainly.

Results:
{results}

Question: {question}
Answer:"""
```

---

## 9. Build order

Build and test one module at a time before moving on. Each step has a concrete pass/fail check.

**Step 1 — Environment**
Start `llama-server.exe` with your model and `-ngl 99`. Then run `python -c "from src import llm_client; print(llm_client.health_check())"` — should print `True`. Check Task Manager while a generation runs: VRAM should spike, not CPU RAM. If VRAM stays flat, `-ngl 99` didn't take effect — check that your llama.cpp build was compiled with CUDA support.

**Step 2 — Embedder**
`embedder.embed("test sentence")` returns a list of 384 floats. Should take under 100ms after the model is warm.

**Step 3 — Vector store**
Add two sentences, search for one, confirm similarity > 0.8 for the right one and < 0.4 for a completely unrelated one.

**Step 4 — Graph store**
`ensure_schema()` runs without error. Insert a dummy Chunk and Entity, create an edge, query it back. Confirm the Kuzu database directory is written to disk.

**Step 5 — Entity extractor**
Pass a sentence like "Fable 5 was developed by Playground Games, a subsidiary of Xbox Game Studios." The output should be a valid `ExtractedEntities` with entities `["Fable 5", "Playground Games", "Xbox Game Studios"]` and at least one relationship tuple. If the LLM breaks the JSON format, the `extract()` function should raise a clear error, not silently return empty data.

**Step 6 — Hybrid retrieval pipeline (no web search yet)**
Seed the store with a few facts using the write-back path directly (bypass the feedback gate for seeding). Ask about one of them. Confirm `source == "local_kb"`. Ask about something not in the store. Confirm `needs_permission == True`.

**Step 7 — Web search**
Call `web_search.search("what is Fable 5", 3)` in isolation. Confirm you get back title + url + snippet for at least 2 results. This is where Tavily API key issues usually surface.

**Step 8 — Full pipeline**
Ask "what is Fable 5?" on an empty store → permission prompt → y → cited web answer → thumbs up → ask same question again → should answer from local KB without web search.

**Step 9 — Terminal REPL**
Run `main.py` and do the full Fable 5 flow interactively. Check that up-arrow history works and that `rich` output renders correctly in your terminal.

---

## 10. Acceptance test

This verifies the core loop end to end:

1. Fresh store (delete `data/` directories)
2. Ask "what is Fable 5?" → expect: "I don't have anything on that. Should I check online? [y/n]"
3. Type `y` → expect: a cited answer with at least one URL
4. Type `u` → expect: "Saved to knowledge base."
5. Ask "what is Fable 5?" again → expect: answer returned immediately from local KB, no web search triggered
6. Now ask something multi-hop: "who developed Fable 5?" — since entities were extracted and linked in the graph during step 4, the graph traversal should pull the Playground Games entity and surface the stored chunk, even though the question wording doesn't match the original Q exactly

Step 6 is what distinguishes the GraphRAG approach from a plain vector store — a cosine match between "who developed Fable 5" and the stored "what is Fable 5 / answer" chunk would be mediocre, but the graph traversal finds it via the Fable 5 entity node.

---

## 11. Phase 2 backlog (don't build yet)

- Reranker (`bge-reranker-base`, CPU) between retrieval and generation
- Query decomposition: break multi-part questions into sub-queries before retrieval
- TTL freshness tags on web-sourced facts with a re-verification prompt
- Document ingestion CLI: chunk + embed + extract entities from a user's own files
- Graph compaction: periodic dedup pass that merges near-duplicate entity nodes
- Richer Kuzu schema: add Fact nodes separate from Chunk nodes for finer granularity
