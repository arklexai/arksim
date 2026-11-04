import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "data" / "reports"
INDEX_PATH = BASE_DIR / "data" / "index.jsonl"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# 建議預設只 ingest 有分析價值的主體部分
START_PAGE = int(os.environ.get("START_PAGE", "3"))   # 1-indexed
END_PAGE = int(os.environ.get("END_PAGE", "40"))      # 1-indexed


def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text_with_metadata(
    text: str,
    chunk_size: int,
    overlap: int,
    base_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            row = dict(base_meta)
            row["chunk_id"] = chunk_id
            row["text"] = chunk_text
            chunks.append(row)

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def infer_ticker_from_filename(filename: str) -> str:
    return filename.split("_")[0].upper()


def main():
    rows = []

    for path in sorted(REPORTS_DIR.glob("*.pdf")):
        reader = PdfReader(str(path))
        ticker = infer_ticker_from_filename(path.name)

        selected_pages = []
        total_pages = len(reader.pages)

        start_idx = max(0, START_PAGE - 1)
        end_idx = min(total_pages, END_PAGE)

        for page_num in range(start_idx, end_idx):
            raw_text = reader.pages[page_num].extract_text() or ""
            text = normalize_text(raw_text)
            if not text:
                continue

            selected_pages.append({
                "page_number": page_num + 1,
                "text": text,
            })

        for page in selected_pages:
            base_meta = {
                "doc": path.name,
                "ticker": ticker,
                "source_type": "pdf",
                "page_start": page["page_number"],
                "page_end": page["page_number"],
            }

            page_chunks = chunk_text_with_metadata(
                text=page["text"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
                base_meta=base_meta,
            )

            for row in page_chunks:
                row["embedding"] = embed_text(row["text"])
                rows.append(row)

    with INDEX_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} chunks to {INDEX_PATH}")


if __name__ == "__main__":
    main()


# import json
# import os
# from pathlib import Path
# from openai import OpenAI
# from dotenv import load_dotenv
# from pypdf import PdfReader

# load_dotenv()

# client = OpenAI()

# BASE_DIR = Path(__file__).resolve().parents[1]
# REPORTS_DIR = BASE_DIR / "data" / "reports"
# INDEX_PATH = BASE_DIR / "data" / "index.jsonl"


# def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
#     return chunks


# def embed_text(text: str) -> list[float]:
#     model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
#     resp = client.embeddings.create(model=model, input=text)
#     return resp.data[0].embedding


# def main():
#     rows = []

#     for path in REPORTS_DIR.glob("*.pdf"):
#         reader = PdfReader(str(path))
#         text = "\n".join(page.extract_text() or "" for page in reader.pages)
#         for i, chunk in enumerate(chunk_text(text)):
#             rows.append({
#                 "doc": path.name,
#                 "chunk_id": i,
#                 "text": chunk,
#                 "embedding": embed_text(chunk),
#             })

#     with INDEX_PATH.open("w", encoding="utf-8") as f:
#         for row in rows:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     print(f"Wrote {len(rows)} chunks to {INDEX_PATH}")


# if __name__ == "__main__":
#     main()