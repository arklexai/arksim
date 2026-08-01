from __future__ import annotations

import re
from typing import Any, Dict, List


FOCUS_KEYWORDS = {
    "margins": [
        "gross margin", "operating margin", "gross profit", "operating income",
        "margin", "profitability", "gaap", "non-gaap", "net income", "cost of revenue"
    ],
    "china_export": [
        "china", "export", "license", "licensing", "h20", "restriction",
        "government", "charge", "regulated", "geopolitical", "3a090", "4a090"
    ],
    "customer_concentration": [
        "customer", "concentration", "customer a", "customer b",
        "direct customer", "revenue concentration", "hyperscale", "cloud provider",
        "accounts receivable"
    ],
    "supply_capacity": [
        "supply", "capacity", "lead time", "backlog", "demand",
        "purchase obligation", "foundry", "constraint", "shipment",
        "osat", "commitment", "commitments", "prepayment", "take-or-pay", "inventory"
    ],
    "blackwell": [
        "blackwell", "hopper", "gpu ramp", "product mix", "platform transition"
    ],
    "risk_factors": [
        "risk", "uncertainty", "competition", "competitive", "regulation",
        "export control", "risk factors", "md&a", "forward-looking"
    ],
    "cash_flow_liquidity": [
        "cash flow", "liquidity", "cash", "marketable securities",
        "capital return", "repurchase", "buyback", "debt", "inventory",
        "operating cash flow", "capital expenditures", "capex"
    ],
    "segment_performance": [
        "data center", "compute", "networking", "graphics", "gaming",
        "professional visualization", "automotive", "segment", "revenue",
        "sequential", "year-over-year", "yoy", "qoq", "operating income"
    ],
    "opex_rnd": [
        "operating expenses", "opex", "r&d", "research and development",
        "sales and marketing", "general and administrative"
    ],
    "file_selection": [
        "which filing", "exact filename", "filing date", "which file", "what file", "filename"
    ],
}


def choose_answer_style(user_query: str) -> str:
    q = user_query.lower()

    if "1-2 sentence" in q or "1–2 sentence" in q or "one- or two-sentence" in q:
        return "one_or_two_sentences"
    if "one sentence" in q or "single sentence" in q:
        return "very_short"
    if "which" in q or "single most" in q or "single strongest" in q or "pick one" in q:
        return "single_choice"
    return "default"


def choose_retrieval_focus(user_query: str) -> str:
    q = user_query.lower()

    if any(k in q for k in ["which filing", "exact filename", "filing date", "which file", "what file", "filename"]):
        return "file_selection"
    if any(k in q for k in ["gross margin", "operating margin", "margin", "gross profit", "gaap", "non-gaap", "net income", "cost of revenue"]):
        return "margins"
    if any(k in q for k in ["china", "export", "license", "h20", "charge", "geopolitical", "3a090", "4a090"]):
        return "china_export"
    if any(k in q for k in ["customer concentration", "customer", "customer a", "customer b", "hyperscale", "cloud provider", "accounts receivable"]):
        return "customer_concentration"
    if any(k in q for k in ["supply", "capacity", "lead time", "backlog", "foundry", "shipment", "osat", "commitment", "prepayment", "take-or-pay", "purchase obligation"]):
        return "supply_capacity"
    if any(k in q for k in ["blackwell", "hopper", "product mix"]):
        return "blackwell"
    if any(k in q for k in ["cash flow", "liquidity", "repurchase", "buyback", "cash", "marketable securities", "debt", "inventory", "capex", "capital expenditures"]):
        return "cash_flow_liquidity"
    if any(k in q for k in ["segment", "data center", "compute", "networking", "graphics", "gaming", "professional visualization", "automotive", "yoy", "qoq", "sequential", "operating income"]):
        return "segment_performance"
    if any(k in q for k in ["operating expenses", "opex", "r&d", "research and development"]):
        return "opex_rnd"
    if any(k in q for k in ["risk", "competition", "competitive", "regulation", "risk factors", "md&a"]):
        return "risk_factors"
    return "default"


def build_targeted_retrieval_query(
    company: str,
    ticker: str,
    goal: str,
    role: str,
    user_query: str,
    retrieval_focus: str,
) -> str:
    extra = " ".join(FOCUS_KEYWORDS.get(retrieval_focus, []))
    return (
        f"{company} {ticker}. Role: {role}. Overall goal: {goal}. "
        f"Current question: {user_query}. "
        f"Focus topics: {extra}"
    )


def rerank_chunks_for_focus(
    chunks: List[Dict[str, Any]],
    user_query: str,
    retrieval_focus: str,
    top_k: int = 7,
) -> List[Dict[str, Any]]:
    keywords = [k.lower() for k in FOCUS_KEYWORDS.get(retrieval_focus, [])]
    q_terms = [t for t in re.findall(r"[a-zA-Z0-9\\-]+", user_query.lower()) if len(t) > 2]

    scored = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).lower()
        score = 0

        for kw in keywords:
            if kw in text:
                score += 4
        for term in q_terms:
            if term in text:
                score += 1

        page = chunk.get("page_start")
        if isinstance(page, int):
            if retrieval_focus in {"margins", "cash_flow_liquidity", "opex_rnd"} and 2 <= page <= 8:
                score += 3
            if retrieval_focus in {"segment_performance", "risk_factors", "china_export", "blackwell", "supply_capacity", "customer_concentration"} and 20 <= page <= 30:
                score += 3

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return chunks[:top_k]

    if all(score == 0 for score, _ in scored[:top_k]):
        return chunks[:top_k]

    return [chunk for _, chunk in scored[:top_k]]


def build_evidence_objects(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = []
    for chunk in chunks:
        evidence.append(
            {
                "source_type": "pdf",
                "title": chunk.get("doc", "unknown_report"),
                "snippet": str(chunk.get("text", "")).strip()[:280],
                "page": chunk.get("page_start"), 
                "url": None,
            }
        )
    return evidence