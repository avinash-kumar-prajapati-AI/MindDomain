import json
from src import llm_client
from src.models import ExtractedEntities

ENTITY_EXTRACTION_PROMPT = """Extract named entities and relationships from the text below.
Respond ONLY with valid JSON matching this exact schema, no other text:
{{
  "entities": ["entity1", "entity2"],
  "relationships": [["subject", "relation", "object"]]
}}

Text:
{text}

JSON:"""

def extract(text: str) -> ExtractedEntities:
    """
    Leverages the Groq LLM in JSON mode to extract semantic entities and relationships.
    Raises ValueError if the output format is invalid or cannot be parsed.
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    
    # Request JSON mode from llm_client
    raw = llm_client.generate(prompt, json_mode=True)
    
    # Clean output fences
    clean = raw.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    
    try:
        data = json.loads(clean)
        if "entities" not in data:
            data["entities"] = []
        if "relationships" not in data:
            data["relationships"] = []
            
        # Standardize relationships into tuples of (subject, relation, object)
        cleaned_relationships = []
        for rel in data.get("relationships", []):
            if isinstance(rel, list) and len(rel) >= 3:
                cleaned_relationships.append((str(rel[0]), str(rel[1]), str(rel[2])))
        data["relationships"] = cleaned_relationships
        
        return ExtractedEntities(**data)
    except Exception as e:
        raise ValueError(
            f"Failed to parse LLM entity extraction response: {e}\nRaw LLM Output:\n{raw}"
        ) from e
