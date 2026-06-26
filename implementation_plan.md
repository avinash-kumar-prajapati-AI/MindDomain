# AliveGraphRAG Implementation Plan

We will implement the remaining modules of the Self-Improving Local GraphRAG Engine using the Groq API.

## Proposed Changes

### Configuration & Infrastructure

We will implement the missing files under `src/` and the root folder:

---

### Prompts

#### [NEW] [prompts.py](file:///d:/_Python/02_AL_ML/02_agents/AliveGraphRAG/src/prompts.py)
- Define `ANSWER_FROM_CONTEXT` template for answering questions from local vector/graph database content.
- Define `WEB_SYNTHESIS` template for synthesizing web search results with inline source citations.

---

### Feedback Log

#### [NEW] [feedback_store.py](file:///d:/_Python/02_AL_ML/02_agents/AliveGraphRAG/src/feedback_store.py)
- Set up a SQLite database at `./logs/interactions.db`.
- Log user queries, generated answers, knowledge sources (`local_kb`, `web`, `none`), and user feedback ratings (`up` or `down`).

---

### Orchestration Pipeline

#### [NEW] [rag_pipeline.py](file:///d:/_Python/02_AL_ML/02_agents/AliveGraphRAG/src/rag_pipeline.py)
- **Stage 1 (Retrieval):** Embed the query, query the Chroma vector store, extract query entities, query the Kuzu graph database for direct and 1-hop mentions, and deduplicate findings.
- **Stage 2 (Confidence):** Fall back to web search if similarity is below the configured `vector_threshold` or if the LLM responds with `NOT_FOUND`.
- **Stage 3 (Feedback & Write-back):** Log user feedback. When feedback is `up` (helpful), extract entities from the combined Q&A text and save it to both Chroma vector store and Kuzu graph database.

---

### Terminal Interface

#### [NEW] [main.py](file:///d:/_Python/02_AL_ML/02_agents/AliveGraphRAG/main.py)
- Run `llm_client.health_check()` at startup to fail-fast.
- Implement an interactive terminal REPL using `prompt_toolkit` (for history recall) and `rich` (for colored output).
- Handle state transitions between:
  - `IDLE` (awaiting a new user question)
  - `AWAITING_PERM` (asking if web search fallback should run)
  - `AWAITING_FB` (asking if the answer was helpful with `u` for up and `d` for down)

---

## Verification Plan

### Automated/Unit Verification
We will run custom script executions to verify:
1. Groq client connection and credentials.
2. Initialization of Chroma and Kuzu schemas.
3. Entity extraction format correctness.

### Manual Verification
1. Run `python main.py` in the `agentAPI` env.
2. Ask a question with an empty database -> ensure permission prompt.
3. Accept web search -> verify output contains citations.
4. Give a Thumbs Up (`u`) -> verify save to vector/graph DB.
5. Ask the same question again -> verify fast, local response without web fallback.
6. Ask a related query -> verify graph retrieval.
