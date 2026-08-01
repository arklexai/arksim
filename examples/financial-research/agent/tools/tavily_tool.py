import os
import requests
from typing import List, Dict, Any


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def format_web_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No recent web results found."
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"[WEB-{i}] {title}\nURL: {url}\n{content}")
    return "\n\n".join(parts)