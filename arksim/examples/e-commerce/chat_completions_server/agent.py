import json
import os
import pickle
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from .utils.embedding_config import EmbeddingConfig, load_embedding
from .utils.llm_config import LLMConfig, load_llm
from .utils.loader import CrawledObject, Loader

app = FastAPI(title="Agent Server")


class FaissRetrieverExecutor:
    def __init__(
        self,
        texts: list[Document],
        index_path: str,
        llm_config: LLMConfig,
    ) -> None:
        self.texts: list[Document] = texts
        self.index_path: str = index_path
        self.embedding_model = load_embedding(
            EmbeddingConfig(embedding_provider=llm_config.llm_provider)
        )
        self.llm = load_llm(llm_config)
        self.retriever = self._init_retriever()

    def _init_retriever(self, **kwargs: dict[str, object]) -> VectorStoreRetriever:
        # initiate FAISS retriever with load/save and batched building
        if os.path.isdir(self.index_path) and len(os.listdir(self.index_path)) > 0:
            docsearch: FAISS = FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            if not self.texts:
                raise ValueError(
                    "No documents available to build the index. Please check your knowledge sources."
                )
            # Build in small batches to avoid large embedding requests
            batch_size = 24
            docsearch = None
            for i in range(0, len(self.texts), batch_size):
                batch = self.texts[i : i + batch_size]
                if docsearch is None:
                    docsearch = FAISS.from_documents(batch, self.embedding_model)
                else:
                    docsearch.add_documents(batch)
            # persist index
            if docsearch is not None:
                os.makedirs(self.index_path, exist_ok=True)
                docsearch.save_local(self.index_path)
        retriever = docsearch.as_retriever(**kwargs)
        return retriever

    async def retrieve_w_score(self, query: str) -> list[tuple[Document, float]]:
        k_value: int = (
            4
            if not self.retriever.search_kwargs.get("k")
            else self.retriever.search_kwargs.get("k")
        )
        docs_and_scores: list[
            tuple[Document, float]
        ] = await self.retriever.vectorstore.asimilarity_search_with_score(
            query, k=k_value
        )
        return docs_and_scores

    @staticmethod
    def load_docs(
        database_path: str,
        llm_config: LLMConfig | None = None,
        index_path: str = "./index",
    ) -> "FaissRetrieverExecutor":
        if llm_config is None:
            llm_config = LLMConfig()
        document_path: str = os.path.join(database_path, "agent_knowledge.pkl")
        index_path_resolved: str = os.path.join(database_path, "index")
        with open(document_path, "rb") as fread:
            documents: list[CrawledObject] = pickle.load(fread)

        # convert CrawledObject to Document with filtering
        documents_langchain = []
        for doc in documents:
            content = getattr(doc, "content", None)
            is_error = getattr(doc, "is_error", False)
            if not content or is_error:
                continue
            documents_langchain.append(
                Document(page_content=content, metadata=doc.metadata)
            )

        return FaissRetrieverExecutor(
            texts=documents_langchain,
            index_path=index_path_resolved,
            llm_config=llm_config,
        )


def build_rag(folder_path: str, rag_docs: list[dict[str, Any]]) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filepath: str = os.path.join(folder_path, "agent_knowledge.pkl")
    loader: Loader = Loader()
    docs: list[Any] = []
    if Path(filepath).exists():
        print(f"Loading existing knowledge from {filepath}")
        print(
            "[Warning] If you are building a new knowledge base, please delete the existing knowledge pickle file and index."
        )
        with open(os.path.join(folder_path, "agent_knowledge.pkl"), "rb") as f:
            docs = pickle.load(f)
    else:
        print("Building new knowledge pickle file and index. This may take a while...")
        for doc in rag_docs:
            source: str = doc.get("source")
            num_docs: int = doc.get("num") if doc.get("num") else 1
            if doc.get("type") == "web":
                num_docs = doc.get("num") if doc.get("num") else 1
                urls: list[str] = loader.get_all_urls(source, num_docs)
                crawled_urls: list[Any] = loader.to_crawled_url_objs(urls)
                docs.extend(crawled_urls)

            elif doc.get("type") == "local":
                # support relative path example: ./data/file.txt
                if source.startswith("./"):
                    source = os.path.join(folder_path, source.lstrip("./"))
                # check if the source is a file or a directory
                file_list = []
                try:
                    if os.path.isfile(source):
                        if source.lower().endswith(".zip"):
                            # Extract zip to a temp dir and process all files inside
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
                                if file.startswith("."):
                                    continue
                                file_list.append(os.path.join(root, file))
                    else:
                        # Path does not exist or is not accessible; skip
                        continue
                except Exception:
                    # If any issue occurs while discovering local files, skip this source
                    continue

                if file_list:
                    docs.extend(loader.to_crawled_local_objs(file_list))

            elif doc.get("type") == "text":
                docs.extend(loader.to_crawled_text([source]))
            else:
                raise ValueError(
                    "type must be one of [web, local, text] and it must be provided"
                )

        chunked_docs: list[Any] = Loader.chunk(docs)
        Loader.save(filepath, chunked_docs)


with open("./examples/e-commerce/knowledge.json") as f:
    knowledge_config = json.load(f)["knowledge"]
documents_dir = os.path.join("./examples/e-commerce")
build_rag(documents_dir, knowledge_config)
vector_db = FaissRetrieverExecutor.load_docs(
    database_path=documents_dir, llm_config=LLMConfig(llm_provider="openai")
)


# Define state for application
class State(TypedDict):
    history: list[dict[str, str]]
    question: str
    context: list[dict[str, Any]]
    answer: str


# build Agent
class Agent:
    def __init__(self, context_id: str | None = None) -> None:
        self.llm = ChatOpenAI(model_name="gpt-5-mini")
        self.system_prompt = "You are a customer service agent. Based on the conversation history and the provided context, generate a helpful and accurate reply to the user. Your response must be concise (no more than 40 words). No prefix (such as 'Assistant:', 'AI: ') is needed. Directly answer the user's question.\n\nContext: {context}"
        self.gen_question_prompt = PromptTemplate(
            template="Given a chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. \nChat History: {history}\n\nQuestion:"
        )
        self.context_id = context_id if context_id else str(uuid.uuid4())
        self.conversation_history: list[dict[str, str]] = []
        self.graph = self.build_graph()

    def session_manager(self, state: State) -> dict[str, Any]:
        self.conversation_history.append({"role": "user", "content": state["question"]})
        return {"history": self.conversation_history}

    def update_history(self, state: State) -> dict[str, Any]:
        self.conversation_history.append(
            {"role": "assistant", "content": state["answer"]}
        )
        return {"history": self.conversation_history}

    def _format_history(self, history: list[dict[str, str]]) -> str:
        formatted_history = ""
        for item in history:
            if item["role"] == "user":
                formatted_history += f"User: {item['content']}\n"
            elif item["role"] == "assistant":
                formatted_history += f"Assistant: {item['content']}\n"
        return formatted_history

    async def retrieve(self, state: State) -> dict[str, Any]:
        # regenerate the question based on the conversation history
        question_response = await self.llm.ainvoke(
            self.gen_question_prompt.invoke(
                {"history": self._format_history(state["history"])}
            )
        )
        # Ensure question is a plain string
        question = getattr(question_response, "content", str(question_response))
        docs_and_score = await vector_db.retrieve_w_score(question)
        retrieved_docs = []
        for doc, score in docs_and_score:
            item = {
                "title": doc.metadata.get("title"),
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "confidence": float(score),
            }
            retrieved_docs.append(item)
        return {"context": retrieved_docs}

    async def generate(self, state: State) -> dict[str, Any]:
        docs_content = "\n\n".join(doc["content"] for doc in state["context"])
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(context=docs_content),
            }
        ]
        messages.extend(state["history"])
        response = await self.llm.ainvoke(messages)
        # Ensure answer is a plain string for JSON serialization
        answer_text = getattr(response, "content", str(response))
        # clean the response
        answer_text = (
            answer_text.strip().replace("Assistant:", "").replace("AI:", "").strip()
        )
        return {"answer": answer_text}

    def build_graph(self) -> CompiledStateGraph[State, Any, Any, Any]:
        graph_builder = StateGraph(State).add_sequence(
            [self.session_manager, self.retrieve, self.generate, self.update_history]
        )
        graph_builder.add_edge(START, "session_manager")
        return graph_builder.compile()

    async def invoke(self, question: str) -> str:
        """
        Process user input and return response, maintaining conversation history.
        """
        result = await self.graph.ainvoke({"question": question})
        return result["answer"]
