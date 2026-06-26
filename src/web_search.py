import os
import warnings
from typing import List

# Suppress ddgs package rename warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

from duckduckgo_search import DDGS
from src.config import config

class SearchResult:
    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

def search_duckduckgo(query: str, max_results: int) -> List[SearchResult]:
    """
    Queries DuckDuckGo web search using the DDGS client.
    """
    import time
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, backend="html", max_results=max_results))
                if raw_results:
                    results = []
                    for r in raw_results:
                        results.append(SearchResult(
                            title=r.get("title", "No Title"),
                            url=r.get("href", ""),
                            snippet=r.get("body", "")
                        ))
                    return results
        except Exception as e:
            print(f"DuckDuckGo search attempt {attempt+1} error: {e}")
        time.sleep(1.5 * (attempt + 1))
    return []

def search_tavily(query: str, max_results: int) -> List[SearchResult]:
    """
    Queries Tavily web search using the TavilyClient.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("Tavily search error: TAVILY_API_KEY environment variable is not set.")
        return []
        
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        raw_results = client.search(query=query, max_results=max_results)
        
        results = []
        for r in raw_results.get("results", []):
            results.append(SearchResult(
                title=r.get("title", "No Title"),
                url=r.get("url", ""),
                snippet=r.get("content", "")
            ))
        return results
    except Exception as e:
        print(f"Tavily search error: {e}")
        return []

def search(query: str, max_results: int) -> List[SearchResult]:
    """
    Performs web search utilizing the configured search provider.
    """
    provider = config.web_search.provider.lower().strip()
    if provider == "tavily":
        return search_tavily(query, max_results)
    else:
        # Default to DuckDuckGo
        return search_duckduckgo(query, max_results)
