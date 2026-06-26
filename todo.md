# Todo List: Self-Improving Local GraphRAG Engine (Groq Integration)

This checklist organizes the build and verification steps for implementing the GraphRAG prototype, adapted to use the **Groq API** instead of a local `llama-server.exe`.

---

## 📅 Setup & Configuration

- [ ] **Create Environment**
  - Use the existing `agentAPI` Conda environment or create a clean one:
    ```bash
    conda activate agentAPI
    ```
- [ ] **Install Dependencies**
  - Install core packages and the `groq` SDK:
    ```bash
    pip install groq chromadb kuzu sentence-transformers tavily-python duckduckgo-search pydantic python-dotenv pyyaml prompt_toolkit rich
    ```
- [ ] **Configure `.env`**
  - Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    ```
- [ ] **Configure `config.yaml`**
  - Create a `config.yaml` using a Groq-supported model (e.g., `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`):
    ```yaml
    llm:
      model: "llama-3.3-70b-versatile" # Or "llama-3.1-8b-instant"
      temperature: 0.2
      max_tokens: 1024

    embedding:
      model: "BAAI/bge-small-en-v1.5"

    retrieval:
      vector_threshold: 0.55
      top_k_vector: 4
      top_k_graph: 3

    web_search:
      provider: "tavily"
      max_results: 5

    storage:
      chroma_path: "./data/chroma_db"
      kuzu_path: "./data/kuzu_db"
      sqlite_path: "./logs/interactions.db"

    dedup:
      similarity_threshold: 0.95
    ```

---

## 🛠️ Step 1: Base Configuration & LLM Client

- [ ] **Implement `src/config.py`**
  - Parse `config.yaml` and initialize environment variables using `pydantic` or `pyyaml` + `python-dotenv`.
- [ ] **Implement `src/models.py`**
  - Define Pydantic models for `Citation`, `RetrievalResult`, `PipelineResponse`, and `ExtractedEntities`.
- [ ] **Implement `src/llm_client.py` (Groq SDK)**
  - Initialize the `Groq` client using the SDK.
  - Implement `generate(prompt: str, json_mode: bool = False) -> str` using `client.chat.completions.create`.
  - Implement a `health_check() -> bool` to verify that `GROQ_API_KEY` is loaded and the client can connect to the Groq API.
  - *Developer Note:* When `json_mode=True`, pass `response_format={"type": "json_object"}` to Groq.

---

## 🧠 Step 2: Storage & Embeddings

- [ ] **Implement `src/embedder.py`**
  - Load `BAAI/bge-small-en-v1.5` onto CPU.
  - Implement `embed(text: str, is_query: bool = False) -> list[float]`.
  - *Important:* Prefix queries with `"Represent this sentence for searching relevant passages: "` and documents with `"Represent this passage for retrieval: "`.
- [ ] **Implement `src/vector_store.py` (ChromaDB)**
  - Initialize persistent Chroma Client pointing at `storage.chroma_path`.
  - Ensure collection uses cosine similarity: `metadata={"hnsw:space": "cosine"}`.
  - Implement functions: `search()`, `add()`, and `is_near_duplicate()`.
- [ ] **Implement `src/graph_store.py` (Kuzu DB)**
  - Implement `ensure_schema()` to create `Entity` nodes, `Chunk` nodes, `MENTIONED_IN` relationships, and `RELATED_TO` relationships.
  - Implement `add_chunk_with_entities()` to write chunks, entities, and edges to Kuzu.
  - Implement `search_by_entity()` executing Cypher matching to pull chunks from graph entity lookup and 1-hop relationships.
  - Implement `get_entity_context()` to construct subgraphs for prompts.

---

## 🔍 Step 3: Entity Extraction & Web Search

- [ ] **Implement `src/entity_extractor.py`**
  - Define `ENTITY_EXTRACTION_PROMPT` instructing the model to output a structured list of entities and relationships.
  - Implement `extract(text: str) -> ExtractedEntities` utilizing `llm_client.generate(..., json_mode=True)`.
- [ ] **Implement `src/web_search.py`**
  - Integrate Tavily (or DuckDuckGo fallback) to fetch web search results when local knowledge fails.

---

## 🔗 Step 4: Hybrid RAG Pipeline & Feedback

- [ ] **Implement `src/prompts.py`**
  - Write templates for `ANSWER_FROM_CONTEXT` and `WEB_SYNTHESIS`.
- [ ] **Implement `src/feedback_store.py`**
  - Set up a simple SQLite logging database under `logs/interactions.db`.
- [ ] **Implement `src/rag_pipeline.py`**
  - **Stage 1 (Retrieval):** Fetch candidates from vector search and graph search (extract entities from query), then deduplicate.
  - **Stage 2 (Confidence):** Fallback to web search if similarity is below `vector_threshold` or if LLM returns `NOT_FOUND`.
  - **Stage 3 (Feedback & Write-back):** If feedback is `up` (helpful), extract entities from the Q&A chunk and store it back into both ChromaDB and Kuzu DB.

---

## 💻 Step 5: Terminal REPL & UI

- [ ] **Implement `main.py`**
  - Run `llm_client.health_check()` at startup to fail-fast.
  - Create a loop tracking states: `IDLE`, `AWAITING_PERM`, and `AWAITING_FB`.
  - Integrate `prompt_toolkit` for recall history and `rich` for terminal markdown layout and colors.

---

## 🧪 Verification & Acceptance Testing

- [ ] **Verify LLM Connectivity**
  - Check that a simple test completion request to Groq executes quickly and successfully.
- [ ] **Verify Schema Creation**
  - Confirm `data/chroma_db/` and `data/kuzu_db/` subfolders are created successfully on first run.
- [ ] **Verify Entity Extraction**
  - Verify that the extractor returns structured JSON conforming to `ExtractedEntities` without parsing errors.
- [ ] **End-to-End Acceptance Test Loop**
  1. Ask query in an empty database (should trigger web search permission prompt).
  2. Accept web search, confirm answer is generated with citations.
  3. Give a Thumbs Up (`u`) to save to database.
  4. Ask the same query again (should return immediately from local knowledge).
  5. Ask a multi-hop query (should trigger graph traversal to resolve entities).
