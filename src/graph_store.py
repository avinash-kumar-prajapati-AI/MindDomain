import os
import kuzu
from typing import List
from src.config import config
from src.models import RetrievalResult, ExtractedEntities

_db = None
_conn = None

def get_connection() -> kuzu.Connection:
    """
    Returns a singleton connection to the persistent Kuzu embedded graph DB.
    """
    global _db, _conn
    if _conn is None:
        path = config.storage.kuzu_path
        # Ensure parent directory of the database file exists
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        _db = kuzu.Database(path)
        _conn = kuzu.Connection(_db)
    return _conn

def ensure_schema() -> None:
    """
    Verifies and initializes the Kuzu graph tables if they do not already exist.
    """
    conn = get_connection()
    
    # Retrieve existing tables
    existing_tables = set()
    try:
        res = conn.execute("CALL SHOW_TABLES() RETURN name")
        while res.has_next():
            existing_tables.add(res.get_next()[0])
    except Exception:
        pass
        
    # Create tables if missing
    if "Entity" not in existing_tables:
        try:
            conn.execute("CREATE NODE TABLE Entity (id STRING, name STRING, type STRING, PRIMARY KEY(id))")
        except Exception as e:
            # Fallback in case table exists but wasn't listed
            if "already exists" not in str(e).lower():
                raise e
                
    if "Chunk" not in existing_tables:
        try:
            conn.execute("CREATE NODE TABLE Chunk (id STRING, text STRING, source STRING, timestamp STRING, PRIMARY KEY(id))")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
                
    if "MENTIONED_IN" not in existing_tables:
        try:
            conn.execute("CREATE REL TABLE MENTIONED_IN (FROM Entity TO Chunk)")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
                
    if "RELATED_TO" not in existing_tables:
        try:
            conn.execute("CREATE REL TABLE RELATED_TO (FROM Entity TO Entity, relation STRING)")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e

def add_chunk_with_entities(chunk_id: str, chunk_text: str, 
                             entities: ExtractedEntities, metadata: dict) -> None:
    """
    Saves a Chunk node, inserts Entity nodes, and binds them via edges.
    """
    conn = get_connection()
    source = metadata.get("source", "unknown")
    timestamp = metadata.get("timestamp", "")
    
    # 1. Create the Chunk node
    conn.execute(
        "CREATE (c:Chunk {id: $id, text: $text, source: $source, timestamp: $timestamp})",
        {"id": chunk_id, "text": chunk_text, "source": source, "timestamp": timestamp}
    )
    
    # 2. Add or merge Entities, and link with MENTIONED_IN
    for entity_name in entities.entities:
        entity_id = entity_name.lower().strip()
        if not entity_id:
            continue
            
        # Check if Entity node exists
        check_ent = conn.execute("MATCH (e:Entity) WHERE e.id = $id RETURN e.id", {"id": entity_id})
        if not check_ent.has_next():
            conn.execute(
                "CREATE (e:Entity {id: $id, name: $name, type: 'Entity'})",
                {"id": entity_id, "name": entity_name}
            )
            
        # Create MENTIONED_IN relation
        check_rel = conn.execute(
            "MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk) WHERE e.id = $eid AND c.id = $cid RETURN r",
            {"eid": entity_id, "cid": chunk_id}
        )
        if not check_rel.has_next():
            conn.execute(
                "MATCH (e:Entity), (c:Chunk) WHERE e.id = $eid AND c.id = $cid CREATE (e)-[:MENTIONED_IN]->(c)",
                {"eid": entity_id, "cid": chunk_id}
            )
            
    # 3. Create RELATED_TO relations from extracted entity tuples
    for relation_tuple in entities.relationships:
        if len(relation_tuple) < 3:
            continue
        subj_name, relation_name, obj_name = relation_tuple
        subj_id = subj_name.lower().strip()
        obj_id = obj_name.lower().strip()
        if not subj_id or not obj_id:
            continue
            
        # Ensure subject and object Entity nodes exist
        for ent_name, ent_id in [(subj_name, subj_id), (obj_name, obj_id)]:
            check_node = conn.execute("MATCH (e:Entity) WHERE e.id = $id RETURN e.id", {"id": ent_id})
            if not check_node.has_next():
                conn.execute(
                    "CREATE (e:Entity {id: $id, name: $name, type: 'Entity'})",
                    {"id": ent_id, "name": ent_name}
                )
                
        # Create RELATED_TO edge
        check_edge = conn.execute(
            "MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity) WHERE e1.id = $sid AND e2.id = $oid AND r.relation = $relation RETURN r",
            {"sid": subj_id, "oid": obj_id, "relation": relation_name}
        )
        if not check_edge.has_next():
            conn.execute(
                "MATCH (e1:Entity), (e2:Entity) WHERE e1.id = $sid AND e2.id = $oid CREATE (e1)-[:RELATED_TO {relation: $relation}]->(e2)",
                {"sid": subj_id, "oid": obj_id, "relation": relation_name}
            )

def search_by_entity(query_entities: List[str], top_k: int) -> List[RetrievalResult]:
    """
    Fuzzy matches query entities, traverses mentions and 1-hop relationships,
    and returns top_k related chunks.
    """
    conn = get_connection()
    if not query_entities:
        return []
        
    entity_ids = [name.lower().strip() for name in query_entities if name.strip()]
    if not entity_ids:
        return []
        
    seen_chunk_ids = set()
    raw_results = []
    
    # Query 1: Direct entity mentioned in Chunk
    try:
        res1 = conn.execute(
            "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk) WHERE e.id IN $entity_ids RETURN DISTINCT c.id, c.text, c.source",
            {"entity_ids": entity_ids}
        )
        while res1.has_next():
            row = res1.get_next()
            chunk_id, text, source = row[0], row[1], row[2]
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                raw_results.append((chunk_id, text, source, 1.0)) # Direct matches priority
    except Exception:
        pass
        
    # Query 2: Indirect 1-hop entity related matches
    try:
        res2 = conn.execute(
            "MATCH (e1:Entity)-[:RELATED_TO]->(e2:Entity)-[:MENTIONED_IN]->(c:Chunk) WHERE e1.id IN $entity_ids RETURN DISTINCT c.id, c.text, c.source",
            {"entity_ids": entity_ids}
        )
        while res2.has_next():
            row = res2.get_next()
            chunk_id, text, source = row[0], row[1], row[2]
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                raw_results.append((chunk_id, text, source, 0.8)) # 1-hop matches priority
    except Exception:
        pass
        
    # Sort by score descending and truncate to top_k
    raw_results.sort(key=lambda x: x[3], reverse=True)
    raw_results = raw_results[:top_k]
    
    ret_results = []
    for chunk_id, text, source, score in raw_results:
        ret_results.append(RetrievalResult(
            text=text,
            metadata={"chunk_id": chunk_id, "source": source, "graph_score": score},
            similarity=score,
            source_store="graph"
        ))
    return ret_results

def get_entity_context(entity_name: str) -> str:
    """
    Extracts one-hop relationships to enrich the prompt context.
    """
    conn = get_connection()
    entity_id = entity_name.lower().strip()
    if not entity_id:
        return ""
        
    query = """
    MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
    WHERE e1.id = $id OR e2.id = $id
    RETURN e1.name, r.relation, e2.name
    """
    
    facts = []
    try:
        res = conn.execute(query, {"id": entity_id})
        while res.has_next():
            row = res.get_next()
            facts.append(f"- {row[0]} {row[1]} {row[2]}")
    except Exception:
        pass
        
    if not facts:
        return ""
    return f"Structured relationships for '{entity_name}':\n" + "\n".join(facts)
