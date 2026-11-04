import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
import os

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_PATH = BASE_DIR / "data" / "index.jsonl"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embed(text: str) -> np.ndarray:
    model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def load_index() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    rows = []
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def retrieve_top_chunks(query: str, top_k: int = 5, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = load_index()
    if not rows:
        return []

    if ticker:
        rows = [row for row in rows if str(row.get("ticker", "")).upper() == ticker.upper()]

    if not rows:
        return []

    q = _embed(query)
    scored = []

    for row in rows:
        text = row.get("text", "").strip()
        emb = row.get("embedding")
        if not text or emb is None:
            continue
        v = np.array(emb, dtype=np.float32)
        score = _cosine(q, v)
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def format_rag_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No filing chunks found."

    parts = []
    for i, c in enumerate(chunks, 1):
        doc = c.get("doc", "unknown_doc")
        page_start = c.get("page_start")
        page_end = c.get("page_end")
        page_label = ""
        if page_start is not None and page_end is not None:
            if page_start == page_end:
                page_label = f" (page {page_start})"
            else:
                page_label = f" (pages {page_start}-{page_end})"

        text = c.get("text", "")
        parts.append(f"[PDF-{i}] {doc}{page_label}\n{text}")

    return "\n\n".join(parts)


def find_report_files(reports_dir: str, ticker: str) -> list[Path]:
    base_dir = Path(__file__).resolve().parents[2]
    report_dir_path = (base_dir / reports_dir).resolve()

    ticker_lower = ticker.lower()
    files = sorted(report_dir_path.glob(f"{ticker_lower}_*.pdf"))
    return files

# import json
# from pathlib import Path
# from typing import List, Dict, Any
# import numpy as np
# from openai import OpenAI
# import os

# client = OpenAI()

# BASE_DIR = Path(__file__).resolve().parents[2]
# INDEX_PATH = BASE_DIR / "data" / "index.jsonl"


# def _cosine(a: np.ndarray, b: np.ndarray) -> float:
#     denom = np.linalg.norm(a) * np.linalg.norm(b)
#     if denom == 0:
#         return 0.0
#     return float(np.dot(a, b) / denom)


# def _embed(text: str) -> np.ndarray:
#     model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
#     resp = client.embeddings.create(model=model, input=text)
#     return np.array(resp.data[0].embedding, dtype=np.float32)


# def load_index() -> List[Dict[str, Any]]:
#     if not INDEX_PATH.exists():
#         return []
#     rows = []
#     for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
#         if line.strip():
#             rows.append(json.loads(line))
#     return rows


# def retrieve_top_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#     rows = load_index()
#     if not rows:
#         return []

#     q = _embed(query)
#     scored = []

#     for row in rows:
#         text = row.get("text", "").strip()
#         emb = row.get("embedding")
#         if not text or emb is None:
#             continue
#         v = np.array(emb, dtype=np.float32)
#         score = _cosine(q, v)
#         scored.append((score, row))

#     scored.sort(key=lambda x: x[0], reverse=True)
#     return [row for _, row in scored[:top_k]]


# def format_rag_context(chunks: List[Dict[str, Any]]) -> str:
#     if not chunks:
#         return "No filing chunks found."
#     parts = []
#     for i, c in enumerate(chunks, 1):
#         doc = c.get("doc", "unknown_doc")
#         text = c.get("text", "")
#         parts.append(f"[PDF-{i}] {doc}\n{text}")
#     return "\n\n".join(parts)


# def find_report_files(reports_dir: str, ticker: str) -> list[Path]:
#     base_dir = Path(__file__).resolve().parents[2]   # examples/financial
#     report_dir_path = (base_dir / reports_dir).resolve()

#     ticker_lower = ticker.lower()
#     files = sorted(report_dir_path.glob(f"{ticker_lower}_*.pdf"))

#     return files