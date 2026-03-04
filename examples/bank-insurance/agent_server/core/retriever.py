# SPDX-License-Identifier: Apache-2.0
"""FAISS-based document retrieval using LangChain OpenAI Embeddings.

Index layout inside the knowledge base directory:
    index/
        index.faiss   — native FAISS flat index
        docs.pkl      — list[dict] of {content, metadata} matching index rows
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from .loader import CrawledObject, Loader

logger = logging.getLogger(__name__)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


class FaissRetriever:
    """FAISS-based retriever backed by the OpenAI Embeddings API.

    On first use the index is built from the provided documents and persisted to
    disk. Subsequent instantiations load the pre-built index automatically.
    """

    def __init__(self, documents: list[dict[str, Any]], index_dir: str) -> None:
        """
        Args:
            documents: Pre-filtered list of dicts with keys ``content`` and ``metadata``.
            index_dir: Directory where the FAISS index and document list are stored.
        """
        self.documents = documents
        self.index_dir = Path(index_dir)
        self.index = self._load_or_build()

    def _load_or_build(self) -> faiss.Index:
        index_file = self.index_dir / "index.faiss"
        docs_file = self.index_dir / "docs.pkl"

        if index_file.exists() and docs_file.exists():
            logger.info("Loading existing FAISS index from %s", self.index_dir)
            index = faiss.read_index(str(index_file))
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)
            return index

        if not self.documents:
            raise ValueError(
                "No documents available to build the index. "
                "Please check your knowledge sources."
            )

        logger.info("Building FAISS index from %d documents...", len(self.documents))
        texts = [d["content"] for d in self.documents]
        embeddings = np.array(_embeddings.embed_documents(texts), dtype=np.float32)
        faiss.normalize_L2(embeddings)

        index: faiss.Index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_file))
        with open(docs_file, "wb") as f:
            pickle.dump(self.documents, f)
        logger.info("FAISS index saved to %s", self.index_dir)
        return index

    async def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Return the top-k most relevant document chunks for a query.

        Args:
            query: Natural language search query.
            k: Number of results to return.

        Returns:
            List of dicts with keys: ``content``, ``title``, ``source``, ``confidence``.
        """
        embedding = np.array(await _embeddings.aembed_query(query), dtype=np.float32)
        vec = embedding.reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, k)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx]
                results.append(
                    {
                        "content": doc["content"],
                        "title": doc.get("metadata", {}).get("title", ""),
                        "source": doc.get("metadata", {}).get("source", ""),
                        "confidence": float(dist),
                    }
                )
        return results

    @classmethod
    def load(cls, database_path: str) -> FaissRetriever:
        """Load a FaissRetriever from a pre-built knowledge base directory.

        Reads ``agent_knowledge.pkl`` produced by :func:`build_rag` and either
        loads the existing FAISS index or builds a new one.

        Args:
            database_path: Path to the knowledge base root directory.

        Returns:
            A ready-to-use FaissRetriever instance.
        """
        pkl_path = Path(database_path) / "agent_knowledge.pkl"
        index_dir = str(Path(database_path) / "index")

        with open(pkl_path, "rb") as f:
            raw_docs: list[CrawledObject] = pickle.load(f)

        documents = [
            {"content": doc.content, "metadata": getattr(doc, "metadata", {})}
            for doc in raw_docs
            if getattr(doc, "content", None) and not getattr(doc, "is_error", False)
        ]
        return cls(documents=documents, index_dir=index_dir)


def build_rag(folder_path: str, rag_docs: list[dict[str, Any]]) -> None:
    """Build and persist the document knowledge base from configured sources.

    Skips rebuilding if ``agent_knowledge.pkl`` already exists. Delete that file
    (and the ``index/`` directory) to trigger a full rebuild.

    Args:
        folder_path: Root directory of the knowledge base.
        rag_docs: List of source configs with keys ``type``, ``source``, and
                  optionally ``num``. Types: ``web``, ``local``, ``text``.
    """
    os.makedirs(folder_path, exist_ok=True)

    filepath = os.path.join(folder_path, "agent_knowledge.pkl")
    loader = Loader()
    docs: list[Any] = []

    if Path(filepath).exists():
        print(f"Loading existing knowledge from {filepath}")
        print(
            "[Warning] Delete the knowledge pickle file and index/ directory "
            "to rebuild the knowledge base from scratch."
        )
        with open(filepath, "rb") as f:
            docs = pickle.load(f)
    else:
        print("Building new knowledge base. This may take a while...")
        for doc in rag_docs:
            source: str = doc.get("source")
            num_docs: int = doc.get("num") or 1

            if doc.get("type") == "web":
                urls = loader.get_all_urls(source, num_docs)
                docs.extend(loader.to_crawled_url_objs(urls))

            elif doc.get("type") == "local":
                if source.startswith("./"):
                    source = os.path.join(folder_path, source.lstrip("./"))
                file_list: list[str] = []
                try:
                    if os.path.isfile(source):
                        if source.lower().endswith(".zip"):
                            with tempfile.TemporaryDirectory() as temp_dir:
                                with zipfile.ZipFile(source, "r") as zip_ref:
                                    zip_ref.extractall(temp_dir)
                                for root, _, files in os.walk(temp_dir):
                                    for file in files:
                                        file_list.append(os.path.join(root, file))
                                if file_list:
                                    docs.extend(loader.to_crawled_local_objs(file_list))
                                    continue
                        else:
                            file_list = [source]
                    elif os.path.isdir(source):
                        for root, _, files in os.walk(source):
                            for file in files:
                                if not file.startswith("."):
                                    file_list.append(os.path.join(root, file))
                    else:
                        continue
                except Exception:
                    continue

                if file_list:
                    docs.extend(loader.to_crawled_local_objs(file_list))

            elif doc.get("type") == "text":
                docs.extend(loader.to_crawled_text([source]))

            else:
                raise ValueError(
                    "type must be one of [web, local, text] and it must be provided"
                )

        chunked_docs = Loader.chunk(docs)
        Loader.save(filepath, chunked_docs)
