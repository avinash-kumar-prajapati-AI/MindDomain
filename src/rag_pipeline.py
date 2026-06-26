import datetime
from src import embedder, vector_store, graph_store, entity_extractor, web_search, prompts, feedback_store, llm_client
from src.models import PipelineResponse, Citation, RetrievalResult, ExtractedEntities

def deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """
    Deduplicates list of RetrievalResult by chunk_id or text similarity.
    """
    seen_ids = set()
    seen_texts = set()
    deduped = []
    for r in results:
        chunk_id = r.metadata.get("chunk_id")
        text_norm = r.text.strip().lower()
        if chunk_id:
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                seen_texts.add(text_norm)
                deduped.append(r)
        else:
            if text_norm not in seen_texts:
                seen_texts.add(text_norm)
                deduped.append(r)
    return deduped

def build_context(query: str) -> tuple[str, list[RetrievalResult]]:
    """
    Stage 1: Entity-aware hybrid retrieval.
    Embeds the query, queries ChromaDB, extracts entities from the query,
    queries KuzuDB, and merges + deduplicates the findings.
    """
    from src.config import config
    query_emb = embedder.embed(query, is_query=True)
    
    # 1. Vector Retrieval
    vector_results = vector_store.search(query_emb, top_k=config.retrieval.top_k_vector)
    
    # 2. Graph Retrieval
    query_entities = []
    try:
        extracted = entity_extractor.extract(query)
        if extracted and extracted.entities:
            query_entities = extracted.entities
    except Exception as e:
        print(f"[dim red]Error extracting entities from query: {e}[/]")
        
    graph_results = []
    if query_entities:
        try:
            graph_results = graph_store.search_by_entity(query_entities, top_k=config.retrieval.top_k_graph)
        except Exception as e:
            print(f"[dim red]Error searching graph database: {e}[/]")
            
    # 3. Merge & Deduplicate
    all_results = deduplicate(vector_results + graph_results)
    
    context = "\n\n---\n\n".join(r.text for r in all_results)
    return context, all_results

def answer(query: str) -> PipelineResponse:
    """
    Stage 2: Confidence Check and Local RAG.
    """
    from src.config import config
    context, results = build_context(query)
    
    # Compute maximum similarity score
    max_sim = max((r.similarity for r in results), default=0.0)
    
    # If no results or below vector similarity threshold
    if not results or max_sim < config.retrieval.vector_threshold:
        return PipelineResponse(
            answer="I don't have anything on that. Should I check online? [y/n]",
            source="none",
            needs_permission=True,
            raw_query=query
        )
        
    try:
        raw_response = llm_client.generate(
            prompts.ANSWER_FROM_CONTEXT.format(context=context, question=query)
        )
        
        if raw_response.strip().upper() == "NOT_FOUND":
            return PipelineResponse(
                answer="My stored knowledge doesn't cover that confidently. Should I check online? [y/n]",
                source="none",
                needs_permission=True,
                raw_query=query
            )
            
        return PipelineResponse(answer=raw_response, source="local_kb", raw_query=query)
    except Exception as e:
        print(f"[dim red]LLM local query error: {e}[/]")
        return PipelineResponse(
            answer="An error occurred while generating answer. Should I check online? [y/n]",
            source="none",
            needs_permission=True,
            raw_query=query
        )

def search_web_and_answer(query: str) -> PipelineResponse:
    """
    Stage 3: Web fallback search and synthesis.
    """
    from src.config import config
    try:
        results = web_search.search(query, config.web_search.max_results)
        if not results:
            return PipelineResponse(
                answer="I could not find any relevant web search results for that query.",
                source="none",
                raw_query=query
            )
            
        numbered = "\n".join(f"[{i+1}] {r.title}: {r.snippet}" for i, r in enumerate(results))
        raw_response = llm_client.generate(prompts.WEB_SYNTHESIS.format(results=numbered, question=query))
        
        citations = [Citation(title=r.title, url=r.url) for r in results]
        return PipelineResponse(answer=raw_response, source="web", citations=citations, raw_query=query)
    except Exception as e:
        print(f"[dim red]Web search/synthesis error: {e}[/]")
        return PipelineResponse(
            answer="Failed to fetch or synthesize web results due to an error.",
            source="none",
            raw_query=query
        )

def submit_feedback(response: PipelineResponse, rating: str) -> None:
    """
    Stage 3 (Write-back): Store feedback and index new knowledge if helpful.
    """
    from src.config import config
    # 1. Log to feedback store SQLite
    try:
        feedback_store.log(response, rating)
    except Exception as e:
        print(f"[dim red]Failed to log feedback: {e}[/]")
        
    # 2. Write back to Vector & Graph databases if user rated helpful
    if rating == "up" and response.source in ("web", "local_kb"):
        chunk_text = f"Q: {response.raw_query}\nA: {response.answer}"
        emb = embedder.embed(chunk_text, is_query=False)
        
        try:
            if not vector_store.is_near_duplicate(emb, config.dedup.similarity_threshold):
                # Save to vector store
                metadata = {
                    "source": response.source,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
                chunk_id = vector_store.add(chunk_text, emb, metadata)
                
                # Extract entities and save to graph store
                try:
                    entities = entity_extractor.extract(chunk_text)
                    graph_store.add_chunk_with_entities(chunk_id, chunk_text, entities, metadata)
                except Exception as e_graph:
                    print(f"[dim red]Failed to extract/store graph entities: {e_graph}[/]")
        except Exception as e_vector:
            print(f"[dim red]Failed to save to vector database: {e_vector}[/]")
