from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = CURRENT_DIR / "tools"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prompts import SYSTEM_PROMPT, build_user_prompt
from rag_tool import find_report_files, retrieve_top_chunks, format_rag_context
from filing_tools import (
    choose_answer_style,
    choose_retrieval_focus,
    build_targeted_retrieval_query,
    rerank_chunks_for_focus,
    build_evidence_objects,
)


class FinancialAdvisorAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI()
        self._chat_id = str(uuid.uuid4())

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        scenario = kwargs.get("scenario", {}) or {}
        parsed = self._parse_scenario(scenario)

        role = parsed["role"]
        company = parsed["company"]
        ticker = parsed["ticker"]
        goal = parsed["goal"]
        reports_dir = parsed["reports_dir"]

        report_files = find_report_files(reports_dir=reports_dir, ticker=ticker)

        # Only answer file-selection questions when user is explicitly asking for filename / filing identity.
        file_selection_response = self._maybe_answer_file_selection_question(
            user_query=user_query,
            report_files=report_files,
        )
        if file_selection_response is not None:
            return json.dumps(file_selection_response, ensure_ascii=False)

        answer_style = choose_answer_style(user_query)
        retrieval_focus = choose_retrieval_focus(user_query)

        retrieval_query = build_targeted_retrieval_query(
            company=company,
            ticker=ticker,
            goal=goal,
            role=role,
            user_query=user_query,
            retrieval_focus=retrieval_focus,
        )

        rag_chunks = retrieve_top_chunks(
            query=retrieval_query,
            top_k=14,
            ticker=ticker,
        )

        rag_chunks = rerank_chunks_for_focus(
            chunks=rag_chunks,
            user_query=user_query,
            retrieval_focus=retrieval_focus,
            top_k=7,
        )

        # Derived-answer path for simple gross-margin calculations when evidence clearly supports it.
        derived_response = self._maybe_answer_derived_margin_question(
            user_query=user_query,
            rag_chunks=rag_chunks,
        )
        if derived_response is not None:
            return json.dumps(derived_response, ensure_ascii=False)

        rag_context = format_rag_context(rag_chunks)

        prompt = build_user_prompt(
            role=role,
            company=company,
            ticker=ticker,
            goal=goal,
            user_query=user_query,
            rag_context=rag_context,
            answer_style=answer_style,
            available_files=[p.name for p in report_files],
        )

        response_json = self._call_llm(prompt=prompt)
        final_output = self._postprocess_output(
            raw_output=response_json,
            rag_chunks=rag_chunks,
            report_files=report_files,
            answer_style=answer_style,
        )

        return json.dumps(final_output, ensure_ascii=False)

    def _parse_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(scenario.get("goal", "")).strip()
        user_profile = str(scenario.get("user_profile", "")).strip()
        agent_context = str(scenario.get("agent_context", "")).strip()
        knowledge = scenario.get("knowledge", []) or []

        company = "NVIDIA"
        ticker = "NVDA"
        reports_dir = "./data/reports"

        for item in knowledge:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip().lower()
            content = str(item.get("content", "")).strip()

            if title == "company":
                company = content
            elif title == "ticker":
                ticker = content.upper()
            elif title == "reports_dir":
                reports_dir = content

        return {
            "role": user_profile or "Financial Research Analyst",
            "company": company,
            "ticker": ticker,
            "goal": goal or agent_context or "Analyze the company using the filing.",
            "reports_dir": reports_dir,
        }

    def _maybe_answer_file_selection_question(
        self,
        user_query: str,
        report_files: List[Path],
    ) -> Optional[Dict[str, Any]]:
        q = user_query.lower()

        direct_file_selection_keywords = [
            "which filing",
            "what filing",
            "which file",
            "what file",
            "exact filename",
            "filename",
            "filing date",
            "what specific filing",
            "what specific nvidia filing",
        ]

        if not any(k in q for k in direct_file_selection_keywords):
            return None

        if not report_files:
            return {
                "answer": "I do not currently see a matched local report file in the configured reports directory.",
                "uncertainty": "No matched report file was found, so I cannot identify a filing or filename from local evidence.",
                "evidence": [],
            }

        file_names = [p.name for p in report_files]

        if len(file_names) == 1:
            fname = file_names[0]
            return {
                "answer": (
                    f"The matched local filing I can access is `{fname}`. "
                    f"I can use that file for the analysis, but I cannot verify recency, filing date, or fiscal quarter from filename alone."
                ),
                "uncertainty": (
                    "I am intentionally not inferring fiscal quarter, filing date, or 'most recent' status from the filename alone."
                ),
                "evidence": [
                    {
                        "source_type": "local_file",
                        "title": fname,
                        "snippet": "Matched local report file in the configured reports directory.",
                        "page": None,
                        "url": None,
                    }
                ],
            }

        joined = ", ".join(f"`{name}`" for name in file_names[:5])
        return {
            "answer": (
                f"I can access these matched local filing files: {joined}. "
                f"I should compare them explicitly before claiming which one is latest or assigning a filing date."
            ),
            "uncertainty": (
                "Multiple matched files are present, so I should not assert a single latest filing without comparing them."
            ),
            "evidence": [
                {
                    "source_type": "local_file",
                    "title": name,
                    "snippet": "Matched local report file in the configured reports directory.",
                    "page": None,
                    "url": None,
                }
                for name in file_names[:3]
            ],
        }

    def _maybe_answer_derived_margin_question(
        self,
        user_query: str,
        rag_chunks: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        q = user_query.lower()
        if "gross margin" not in q:
            return None

        combined_text = "\n".join(str(c.get("text", "")) for c in rag_chunks[:5])

        current_match = re.search(
            r"Revenue\s*\$?\s*([0-9,]+).*?Cost of revenue\s*([0-9,]+).*?Gross profit\s*([0-9,]+)",
            combined_text,
            re.IGNORECASE | re.DOTALL,
        )
        if not current_match:
            return None

        try:
            revenue_cur = float(current_match.group(1).replace(",", ""))
            cost_cur = float(current_match.group(2).replace(",", ""))
            gross_cur = float(current_match.group(3).replace(",", ""))
        except Exception:
            return None

        # Try to find prior-year numbers in same chunk sequence.
        trailing = combined_text[current_match.end():]
        prior_match = re.search(
            r"Revenue\s*\$?\s*([0-9,]+).*?Cost of revenue\s*([0-9,]+).*?Gross profit\s*([0-9,]+)",
            trailing,
            re.IGNORECASE | re.DOTALL,
        )
        if not prior_match:
            return None

        try:
            revenue_prev = float(prior_match.group(1).replace(",", ""))
            cost_prev = float(prior_match.group(2).replace(",", ""))
            gross_prev = float(prior_match.group(3).replace(",", ""))
        except Exception:
            return None

        if revenue_cur <= 0 or revenue_prev <= 0:
            return None

        gm_cur = gross_cur / revenue_cur * 100.0
        gm_prev = gross_prev / revenue_prev * 100.0
        delta = gm_cur - gm_prev

        direction = "decrease" if delta < 0 else "increase"
        delta_abs = abs(delta)

        answer = (
            f"Using the retrieved income statement figures, the implied current-quarter gross margin is about {gm_cur:.1f}% "
            f"and the prior-year-quarter gross margin is about {gm_prev:.1f}%. That is a {delta_abs:.1f} percentage-point {direction}. "
            f"Based only on these excerpts, this supports a directional read on margin change, but not a filing-grounded claim about the exact driver unless management explicitly states one elsewhere in the retrieved excerpts."
        )

        return {
            "answer": answer,
            "uncertainty": "The retrieved excerpts support the calculation, but not necessarily a specific causal explanation for the margin move.",
            "evidence": build_evidence_objects(rag_chunks[:2]),
        }

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "financial_agent_turn_output",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "uncertainty": {"type": ["string", "null"]},
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_type": {"type": "string"},
                                        "title": {"type": "string"},
                                        "snippet": {"type": "string"},
                                        "page": {"type": ["integer", "null"]},
                                        "url": {"type": ["string", "null"]},
                                    },
                                    "required": ["source_type", "title", "snippet", "page", "url"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["answer", "uncertainty", "evidence"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        return json.loads(response.output_text)

    def _postprocess_output(
        self,
        raw_output: Dict[str, Any],
        rag_chunks: List[Dict[str, Any]],
        report_files: List[Path],
        answer_style: str,
    ) -> Dict[str, Any]:
        answer = str(raw_output.get("answer", "")).strip()
        uncertainty = raw_output.get("uncertainty")

        if not answer:
            answer = "I could not form a grounded answer from the retrieved filing evidence."

        answer = self._normalize_answer_length(answer, answer_style)
        answer = self._sanitize_answer(answer)

        evidence = raw_output.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        cleaned_evidence = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            cleaned_evidence.append(
                {
                    "source_type": str(item.get("source_type", "pdf")),
                    "title": self._sanitize_title(str(item.get("title", ""))),
                    "snippet": str(item.get("snippet", ""))[:280],
                    "page": item.get("page"),
                    "url": item.get("url"),
                }
            )

        evidence = cleaned_evidence

        if len(evidence) < 2:
            evidence = build_evidence_objects(rag_chunks[:3])

        if not evidence and report_files:
            evidence = [
                {
                    "source_type": "pdf",
                    "title": p.name,
                    "snippet": "Matched report file used for retrieval.",
                    "page": None,
                    "url": None,
                }
                for p in report_files[:2]
            ]

        return {
            "answer": answer,
            "uncertainty": uncertainty,
            "evidence": evidence,
        }

    def _normalize_answer_length(self, answer: str, answer_style: str) -> str:
        text = re.sub(r"\s+", " ", answer).strip()

        if answer_style == "one_or_two_sentences":
            sentences = re.split(r"(?<=[.!?])\s+", text)
            text = " ".join(sentences[:2]).strip()
        elif answer_style == "single_choice":
            sentences = re.split(r"(?<=[.!?])\s+", text)
            text = " ".join(sentences[:3]).strip()
        elif answer_style == "very_short":
            sentences = re.split(r"(?<=[.!?])\s+", text)
            text = " ".join(sentences[:1]).strip()

        return text

    def _sanitize_answer(self, answer: str) -> str:
        text = answer

        dangerous_patterns = [
            r"\bFY20\d{2}\s*Q[1-4]\b",
            r"\bfiscal year\s*20\d{2}\b",
            r"\bfiled on [A-Z][a-z]+ \d{1,2}, \d{4}\b",
            r"\bmost recent quarter'?s filing\b",
            r"\blatest quarter'?s filing\b",
            r"\bquarter ended [A-Z][a-z]+ \d{1,2}, \d{4}\b",
        ]
        for pat in dangerous_patterns:
            text = re.sub(pat, "the matched filing", text, flags=re.IGNORECASE)

        return text.strip()

    def _sanitize_title(self, title: str) -> str:
        if not title:
            return "matched_filing"
        return title.strip()
    

    
# from __future__ import annotations

# import json
# import os
# import sys
# import uuid
# from pathlib import Path
# from typing import Any, Dict, List

# from dotenv import load_dotenv
# from openai import OpenAI

# from arksim.config import AgentConfig
# from arksim.simulation_engine.agent.base import BaseAgent

# load_dotenv()

# CURRENT_DIR = Path(__file__).resolve().parent
# TOOLS_DIR = CURRENT_DIR / "tools"

# if str(CURRENT_DIR) not in sys.path:
#     sys.path.insert(0, str(CURRENT_DIR))
# if str(TOOLS_DIR) not in sys.path:
#     sys.path.insert(0, str(TOOLS_DIR))

# from prompts import SYSTEM_PROMPT, build_user_prompt
# from rag_tool import find_report_files, retrieve_top_chunks, format_rag_context




# class FinancialAdvisorAgent(BaseAgent):
#     def __init__(self, agent_config: AgentConfig) -> None:
#         super().__init__(agent_config)
#         self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
#         self.client = OpenAI()
#         self._chat_id = str(uuid.uuid4())

#     async def get_chat_id(self) -> str:
#         return self._chat_id

#     async def execute(self, user_query: str, **kwargs: object) -> str:
#         scenario = kwargs.get("scenario", {}) or {}
#         parsed = self._parse_scenario(scenario)

#         role = parsed["role"]
#         company = parsed["company"]
#         ticker = parsed["ticker"]
#         goal = parsed["goal"]
#         constraints = parsed["constraints"]
#         reports_dir = parsed["reports_dir"]

#         report_files = find_report_files(reports_dir=reports_dir, ticker=ticker)

#         retrieval_query = self._build_retrieval_query(
#             company=company,
#             ticker=ticker,
#             goal=goal,
#             role=role,
#             constraints=constraints + [f"user_query: {user_query}"],
#         )

#         rag_chunks = retrieve_top_chunks(
#             query=retrieval_query,
#             top_k=5,
#             ticker=ticker,
#         )
#         rag_context = format_rag_context(rag_chunks)

#         prompt = build_user_prompt(
#             role=role,
#             company=company,
#             ticker=ticker,
#             goal=goal,
#             constraints=constraints + [f"Current user query: {user_query}"],
#             rag_context=rag_context,
#         )

#         response_json = self._call_llm(prompt=prompt, report_files=report_files)
#         final_output = self._postprocess_output(
#             raw_output=response_json,
#             rag_chunks=rag_chunks,
#             report_files=report_files,
#         )

#         return json.dumps(final_output, ensure_ascii=False)

#     def _parse_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
#         goal = str(scenario.get("goal", "")).strip()
#         user_profile = str(scenario.get("user_profile", "")).strip()
#         agent_context = str(scenario.get("agent_context", "")).strip()
#         knowledge = scenario.get("knowledge", []) or []

#         company = "NVIDIA"
#         ticker = "NVDA"
#         reports_dir = "./data/reports"

#         for item in knowledge:
#             if not isinstance(item, dict):
#                 continue
#             title = str(item.get("title", "")).strip().lower()
#             content = str(item.get("content", "")).strip()

#             if title == "company":
#                 company = content
#             elif title == "ticker":
#                 ticker = content.upper()
#             elif title == "reports_dir":
#                 reports_dir = content

#         return {
#             "role": user_profile or "Financial Research Analyst",
#             "company": company,
#             "ticker": ticker,
#             "goal": goal or agent_context or "Analyze the company using the filing.",
#             "constraints": [],
#             "reports_dir": reports_dir,
#         }

#     def _build_retrieval_query(
#         self,
#         company: str,
#         ticker: str,
#         goal: str,
#         role: str,
#         constraints: List[str],
#     ) -> str:
#         constraints_text = " | ".join(constraints) if constraints else ""
#         return (
#             f"{company} {ticker} {goal}. "
#             f"Role: {role}. "
#             f"Focus on revenue, margins, profitability, cash flow, liquidity, customer concentration, "
#             f"segment performance, risk factors, export controls, inventory, capital return. "
#             f"Constraints: {constraints_text}"
#         )

#     def _call_llm(self, prompt: str, report_files: List[Path]) -> Dict[str, Any]:
#         report_file_names = [p.name for p in report_files]

#         response = self.client.responses.create(
#             model=self.model,
#             input=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {
#                     "role": "user",
#                     "content": (
#                         f"{prompt}\n\n"
#                         f"Matched report files in reports directory: {report_file_names}\n\n"
#                         "Use only the evidence provided. "
#                         "Do not fabricate exact figures if absent. "
#                         "This is research simulation output, not personalized investment advice."
#                     ),
#                 },
#             ],
#             text={
#                 "format": {
#                     "type": "json_schema",
#                     "name": "financial_agent_output",
#                     "schema": {
#                         "type": "object",
#                         "properties": {
#                             "summary": {"type": "string"},
#                             "bull_case": {"type": "array", "items": {"type": "string"}},
#                             "bear_case": {"type": "array", "items": {"type": "string"}},
#                             "key_risks": {"type": "array", "items": {"type": "string"}},
#                             "conclusion": {"type": "string"},
#                             "confidence": {"type": "string"},
#                             "sources": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "object",
#                                     "properties": {
#                                         "source_type": {"type": "string"},
#                                         "title": {"type": "string"},
#                                         "snippet": {"type": "string"},
#                                         "url": {"type": ["string", "null"]},
#                                     },
#                                     "required": ["source_type", "title", "snippet", "url"],
#                                     "additionalProperties": False,
#                                 },
#                             },
#                         },
#                         "required": [
#                             "summary",
#                             "bull_case",
#                             "bear_case",
#                             "key_risks",
#                             "conclusion",
#                             "confidence",
#                             "sources",
#                         ],
#                         "additionalProperties": False,
#                     },
#                 }
#             },
#         )
#         return json.loads(response.output_text)

#     def _postprocess_output(
#         self,
#         raw_output: Dict[str, Any],
#         rag_chunks: List[Dict[str, Any]],
#         report_files: List[Path],
#     ) -> Dict[str, Any]:
#         output = {
#             "summary": raw_output.get("summary", ""),
#             "bull_case": raw_output.get("bull_case", []),
#             "bear_case": raw_output.get("bear_case", []),
#             "key_risks": raw_output.get("key_risks", []),
#             "conclusion": raw_output.get("conclusion", ""),
#             "confidence": raw_output.get("confidence", "low"),
#             "sources": raw_output.get("sources", []),
#         }

#         if len(output["sources"]) < 2:
#             sources = []
#             for chunk in rag_chunks[:3]:
#                 sources.append({
#                     "source_type": "pdf",
#                     "title": chunk.get("doc", "unknown_report"),
#                     "snippet": chunk.get("text", "")[:300].strip(),
#                     "url": None,
#                 })

#             if not sources:
#                 for p in report_files[:2]:
#                     sources.append({
#                         "source_type": "pdf",
#                         "title": p.name,
#                         "snippet": "Matched report file used for retrieval.",
#                         "url": None,
#                     })

#             output["sources"] = sources

#         return output




# from __future__ import annotations

# import json
# import os
# import sys
# import uuid
# from pathlib import Path
# from typing import Any, Dict, List

# from dotenv import load_dotenv
# from openai import OpenAI

# from arksim.config import AgentConfig
# from arksim.simulation_engine.agent.base import BaseAgent

# load_dotenv()

# CURRENT_DIR = Path(__file__).resolve().parent
# TOOLS_DIR = CURRENT_DIR / "tools"

# if str(CURRENT_DIR) not in sys.path:
#     sys.path.insert(0, str(CURRENT_DIR))
# if str(TOOLS_DIR) not in sys.path:
#     sys.path.insert(0, str(TOOLS_DIR))

# from prompts import SYSTEM_PROMPT, build_user_prompt
# from rag_tool import find_report_files, retrieve_top_chunks, format_rag_context
# from tavily_tool import tavily_search, format_web_context
# from financial_data_tool import build_financial_context


# class FinancialAdvisorAgent(BaseAgent):
#     def __init__(self, agent_config: AgentConfig) -> None:
#         super().__init__(agent_config)
#         self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
#         self.client = OpenAI()
#         self._chat_id = str(uuid.uuid4())

#     async def get_chat_id(self) -> str:
#         return self._chat_id

#     async def execute(self, user_query: str, **kwargs: object) -> str:
#         scenario = kwargs.get("scenario", {}) or {}
#         parsed = self._parse_scenario(scenario)

#         role = parsed["role"]
#         company = parsed["company"]
#         ticker = parsed["ticker"]
#         goal = parsed["goal"]
#         constraints = parsed["constraints"]
#         reports_dir = parsed["reports_dir"]
#         do_web_search = parsed["web_search"]

#         report_files = find_report_files(reports_dir=reports_dir, ticker=ticker)

#         retrieval_query = self._build_retrieval_query(
#             company=company,
#             ticker=ticker,
#             goal=goal,
#             role=role,
#             constraints=constraints + [f"user_query: {user_query}"],
#         )

#         rag_chunks = retrieve_top_chunks(query=retrieval_query, top_k=5)
#         rag_context = format_rag_context(rag_chunks)
#         financial_context = build_financial_context(ticker=ticker)

#         web_results: List[Dict[str, Any]] = []
#         if do_web_search:
#             web_results = tavily_search(
#                 query=f"{company} {ticker} {user_query} latest earnings news risks catalysts guidance demand margins",
#                 max_results=5,
#             )
#         web_context = format_web_context(web_results)

#         prompt = build_user_prompt(
#             role=role,
#             company=company,
#             ticker=ticker,
#             goal=goal,
#             constraints=constraints + [f"Current user query: {user_query}"],
#             rag_context=rag_context,
#             financial_context=financial_context,
#             web_context=web_context,
#         )

#         response_json = self._call_llm(prompt=prompt, report_files=report_files)
#         final_output = self._postprocess_output(
#             raw_output=response_json,
#             rag_chunks=rag_chunks,
#             web_results=web_results,
#             report_files=report_files,
#         )

#         return json.dumps(final_output, ensure_ascii=False)

#     def _parse_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
#         goal = str(scenario.get("goal", "")).strip()
#         user_profile = str(scenario.get("user_profile", "")).strip()
#         agent_context = str(scenario.get("agent_context", "")).strip()
#         knowledge = scenario.get("knowledge", []) or []

#         company = "NVIDIA"
#         ticker = "NVDA"
#         reports_dir = "./data/reports"
#         web_search = True

#         for item in knowledge:
#             if not isinstance(item, dict):
#                 continue
#             title = str(item.get("title", "")).strip().lower()
#             content = str(item.get("content", "")).strip()

#             if title == "company":
#                 company = content
#             elif title == "ticker":
#                 ticker = content.upper()
#             elif title == "reports_dir":
#                 reports_dir = content
#             elif title == "use_recent_news":
#                 web_search = content.lower() == "true"

#         return {
#             "role": user_profile or "Financial Research Analyst",
#             "company": company,
#             "ticker": ticker,
#             "goal": goal or agent_context or "Analyze the company.",
#             "constraints": [],
#             "reports_dir": reports_dir,
#             "web_search": web_search,
#             "agent_context": agent_context,
#         }

#     def _build_retrieval_query(
#         self,
#         company: str,
#         ticker: str,
#         goal: str,
#         role: str,
#         constraints: List[str],
#     ) -> str:
#         constraints_text = " | ".join(constraints) if constraints else ""
#         return (
#             f"{company} {ticker} {goal}. "
#             f"Role: {role}. "
#             f"Focus on revenue, margins, guidance, demand, risks, competition, "
#             f"capital expenditure, cash flow, customer concentration, regulatory risks. "
#             f"Constraints: {constraints_text}"
#         )

#     def _call_llm(self, prompt: str, report_files: List[Path]) -> Dict[str, Any]:
#         report_file_names = [p.name for p in report_files]

#         response = self.client.responses.create(
#             model=self.model,
#             input=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {
#                     "role": "user",
#                     "content": (
#                         f"{prompt}\n\n"
#                         f"Matched report files in reports directory: {report_file_names}\n\n"
#                         "Use only the evidence provided. "
#                         "Do not fabricate exact figures if absent. "
#                         "This is research simulation output, not personalized investment advice."
#                     ),
#                 },
#             ],
#             text={
#                 "format": {
#                     "type": "json_schema",
#                     "name": "financial_agent_output",
#                     "schema": {
#                         "type": "object",
#                         "properties": {
#                             "summary": {"type": "string"},
#                             "bull_case": {"type": "array", "items": {"type": "string"}},
#                             "bear_case": {"type": "array", "items": {"type": "string"}},
#                             "key_risks": {"type": "array", "items": {"type": "string"}},
#                             "conclusion": {"type": "string"},
#                             "confidence": {"type": "string"},
#                             "sources": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "object",
#                                     "properties": {
#                                         "source_type": {"type": "string"},
#                                         "title": {"type": "string"},
#                                         "snippet": {"type": "string"},
#                                         "url": {"type": ["string", "null"]},
#                                     },
#                                     "required": ["source_type", "title", "snippet", "url"],
#                                     "additionalProperties": False,
#                                 },
#                             },
#                         },
#                         "required": ["summary", "bull_case", "bear_case", "key_risks", "conclusion", "confidence", "sources"],
#                         "additionalProperties": False,
#                     },
#                 }
#             },
#         )
#         return json.loads(response.output_text)

#     def _postprocess_output(
#         self,
#         raw_output: Dict[str, Any],
#         rag_chunks: List[Dict[str, Any]],
#         web_results: List[Dict[str, Any]],
#         report_files: List[Path],
#     ) -> Dict[str, Any]:
#         output = {
#             "summary": raw_output.get("summary", ""),
#             "bull_case": raw_output.get("bull_case", []),
#             "bear_case": raw_output.get("bear_case", []),
#             "key_risks": raw_output.get("key_risks", []),
#             "conclusion": raw_output.get("conclusion", ""),
#             "confidence": raw_output.get("confidence", "low"),
#             "sources": raw_output.get("sources", []),
#         }

#         if len(output["sources"]) < 2:
#             sources = []
#             for chunk in rag_chunks[:3]:
#                 sources.append({
#                     "source_type": "pdf",
#                     "title": chunk.get("doc", "unknown_report"),
#                     "snippet": chunk.get("text", "")[:300].strip(),
#                     "url": None,
#                 })
#             for item in web_results[:2]:
#                 sources.append({
#                     "source_type": "web",
#                     "title": item.get("title", "web_result"),
#                     "snippet": item.get("content", "")[:300].strip(),
#                     "url": item.get("url"),
#                 })
#             if not sources:
#                 for p in report_files[:2]:
#                     sources.append({
#                         "source_type": "pdf",
#                         "title": p.name,
#                         "snippet": "Matched report file used for retrieval.",
#                         "url": None,
#                     })
#             output["sources"] = sources

#         return output







# from __future__ import annotations

# import json
# import os
# import sys
# from pathlib import Path
# from typing import Any, Dict, List

# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# CURRENT_DIR = Path(__file__).resolve().parent
# TOOLS_DIR = CURRENT_DIR / "tools"

# if str(CURRENT_DIR) not in sys.path:
#     sys.path.insert(0, str(CURRENT_DIR))
# if str(TOOLS_DIR) not in sys.path:
#     sys.path.insert(0, str(TOOLS_DIR))

# from prompts import SYSTEM_PROMPT, build_user_prompt
# from rag_tool import find_report_files, retrieve_top_chunks, format_rag_context
# from tavily_tool import tavily_search, format_web_context
# from financial_data_tool import build_financial_context


# # from __future__ import annotations

# # import json
# # import os
# # from pathlib import Path
# # from typing import Any, Dict, List

# # from openai import OpenAI

# # from .prompts import SYSTEM_PROMPT, build_user_prompt
# # from .tools.rag_tool import (
# #     find_report_files,
# #     retrieve_top_chunks,
# #     format_rag_context,
# # )
# # from .tools.tavily_tool import tavily_search, format_web_context
# # from .tools.financial_data_tool import build_financial_context

# # from dotenv import load_dotenv
# # load_dotenv()



# class FinancialAdvisorAgent:
#     """
#     Custom ArkSim agent for financial research simulation.

#     Expected scenario shape:
#     {
#       "scenario_id": "...",
#       "user_profile": {
#         "role": "Hedge Fund Analyst",
#         "style": "direct, skeptical"
#       },
#       "task": {
#         "company": "NVIDIA",
#         "ticker": "NVDA",
#         "goal": "Assess near-term upside/downside after latest earnings",
#         "constraints": ["Use filings", "Use recent news"]
#       },
#       "context": {
#         "reports_dir": "./data/reports",
#         "web_search": true
#       }
#     }
#     """

#     def __init__(self, **kwargs: Any) -> None:
#         self.model = kwargs.get("model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
#         self.client = OpenAI()

#     def run(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Main entrypoint used by local scripts / ArkSim custom agent wrapper.
#         Returns a JSON-serializable dict.
#         """
#         parsed = self._parse_scenario(scenario)

#         role = parsed["role"]
#         company = parsed["company"]
#         ticker = parsed["ticker"]
#         goal = parsed["goal"]
#         constraints = parsed["constraints"]
#         reports_dir = parsed["reports_dir"]
#         do_web_search = parsed["web_search"]

#         report_files = find_report_files(reports_dir=reports_dir, ticker=ticker)

#         retrieval_query = self._build_retrieval_query(
#             company=company,
#             ticker=ticker,
#             goal=goal,
#             role=role,
#             constraints=constraints,
#         )

#         # rag_chunks = retrieve_top_chunks(
#         #     query=retrieval_query,
#         #     top_k=5,
#         #     ticker=ticker,
#         # )

#         rag_chunks = retrieve_top_chunks(
#             query=retrieval_query,
#             top_k=5,
#         )

#         rag_context = format_rag_context(rag_chunks)

#         financial_context = build_financial_context(ticker=ticker)

#         web_results: List[Dict[str, Any]] = []
#         if do_web_search:
#             web_results = tavily_search(
#                 query=f"{company} {ticker} latest earnings news risks catalysts guidance demand margins",
#                 max_results=5,
#             )
#         web_context = format_web_context(web_results)

#         prompt = build_user_prompt(
#             role=role,
#             company=company,
#             ticker=ticker,
#             goal=goal,
#             constraints=constraints,
#             rag_context=rag_context,
#             financial_context=financial_context,
#             web_context=web_context,
#         )

#         response_json = self._call_llm(
#             prompt=prompt,
#             report_files=report_files,
#         )

#         final_output = self._postprocess_output(
#             raw_output=response_json,
#             rag_chunks=rag_chunks,
#             web_results=web_results,
#             report_files=report_files,
#         )
#         return final_output

#     # def _parse_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
#     #     user_profile = scenario.get("user_profile", {})
#     #     task = scenario.get("task", {})
#     #     context = scenario.get("context", {})

#     #     role = user_profile.get("role", "Financial Research Analyst")
#     #     company = task.get("company", "").strip()
#     #     ticker = task.get("ticker", "").strip().upper()
#     #     goal = task.get("goal", "").strip()
#     #     constraints = task.get("constraints", [])

#     #     if not company:
#     #         raise ValueError("Scenario missing task.company")
#     #     if not ticker:
#     #         raise ValueError("Scenario missing task.ticker")
#     #     if not goal:
#     #         raise ValueError("Scenario missing task.goal")
#     #     if not isinstance(constraints, list):
#     #         raise ValueError("Scenario task.constraints must be a list")

#     #     reports_dir = context.get("reports_dir", "./data/reports")
#     #     web_search = bool(context.get("web_search", True))

#     #     return {
#     #         "role": role,
#     #         "company": company,
#     #         "ticker": ticker,
#     #         "goal": goal,
#     #         "constraints": constraints,
#     #         "reports_dir": reports_dir,
#     #         "web_search": web_search,
#     #     }


#     def _parse_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
#         goal = scenario.get("goal", "").strip()
#         user_profile = scenario.get("user_profile", "")
#         agent_context = scenario.get("agent_context", "")
#         knowledge = scenario.get("knowledge", [])

#         company = "NVIDIA"
#         ticker = "NVDA"
#         reports_dir = "./data/reports"
#         web_search = True

#         for item in knowledge:
#             title = str(item.get("title", "")).strip().lower()
#             content = str(item.get("content", "")).strip()

#             if title == "company":
#                 company = content
#             elif title == "ticker":
#                 ticker = content.upper()
#             elif title == "reports_dir":
#                 reports_dir = content
#             elif title == "use_recent_news":
#                 web_search = content.lower() == "true"

#         return {
#             "role": user_profile or "Financial Research Analyst",
#             "company": company,
#             "ticker": ticker,
#             "goal": goal,
#             "constraints": [],
#             "reports_dir": reports_dir,
#             "web_search": web_search,
#             "agent_context": agent_context,
#         }

#     def _build_retrieval_query(
#         self,
#         company: str,
#         ticker: str,
#         goal: str,
#         role: str,
#         constraints: List[str],
#     ) -> str:
#         constraints_text = " | ".join(constraints) if constraints else ""
#         return (
#             f"{company} {ticker} {goal}. "
#             f"Role: {role}. "
#             f"Focus on revenue, margins, guidance, demand, risks, competition, "
#             f"capital expenditure, cash flow, customer concentration, regulatory risks. "
#             f"Constraints: {constraints_text}"
#         )

#     def _call_llm(self, prompt: str, report_files: List[Path]) -> Dict[str, Any]:
#         """
#         Calls OpenAI with a strict JSON schema.
#         """
#         report_file_names = [p.name for p in report_files]

#         response = self.client.responses.create(
#             model=self.model,
#             input=[
#                 {
#                     "role": "system",
#                     "content": SYSTEM_PROMPT,
#                 },
#                 {
#                     "role": "user",
#                     "content": (
#                         f"{prompt}\n\n"
#                         f"Matched report files in reports directory: {report_file_names}\n\n"
#                         "Important:\n"
#                         "- Use only the evidence provided.\n"
#                         "- Do not fabricate exact figures if absent.\n"
#                         "- This is research simulation output, not personalized investment advice.\n"
#                         "- Be explicit about uncertainty.\n"
#                     ),
#                 },
#             ],
#             text={
#                 "format": {
#                     "type": "json_schema",
#                     "name": "financial_agent_output",
#                     "schema": {
#                         "type": "object",
#                         "properties": {
#                             "summary": {"type": "string"},
#                             "bull_case": {
#                                 "type": "array",
#                                 "items": {"type": "string"},
#                             },
#                             "bear_case": {
#                                 "type": "array",
#                                 "items": {"type": "string"},
#                             },
#                             "key_risks": {
#                                 "type": "array",
#                                 "items": {"type": "string"},
#                             },
#                             "conclusion": {"type": "string"},
#                             "confidence": {"type": "string"},
#                             "sources": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "object",
#                                     "properties": {
#                                         "source_type": {"type": "string"},
#                                         "title": {"type": "string"},
#                                         "snippet": {"type": "string"},
#                                         "url": {"type": ["string", "null"]},
#                                     },
#                                     "required": ["source_type", "title", "snippet", "url"],
#                                     "additionalProperties": False,
#                                 },
#                             },
#                         },
#                         "required": [
#                             "summary",
#                             "bull_case",
#                             "bear_case",
#                             "key_risks",
#                             "conclusion",
#                             "confidence",
#                             "sources",
#                         ],
#                         "additionalProperties": False,
#                     },
#                 }
#             },
#         )

#         try:
#             return json.loads(response.output_text)
#         except json.JSONDecodeError as exc:
#             raise ValueError(f"Model did not return valid JSON: {exc}") from exc

#     def _postprocess_output(
#         self,
#         raw_output: Dict[str, Any],
#         rag_chunks: List[Dict[str, Any]],
#         web_results: List[Dict[str, Any]],
#         report_files: List[Path],
#     ) -> Dict[str, Any]:
#         """
#         Ensures required fields exist and optionally appends grounded sources
#         if model returned too few / none.
#         """
#         output = {
#             "summary": raw_output.get("summary", ""),
#             "bull_case": raw_output.get("bull_case", []),
#             "bear_case": raw_output.get("bear_case", []),
#             "key_risks": raw_output.get("key_risks", []),
#             "conclusion": raw_output.get("conclusion", ""),
#             "confidence": raw_output.get("confidence", "low"),
#             "sources": raw_output.get("sources", []),
#         }

#         if not isinstance(output["bull_case"], list):
#             output["bull_case"] = []
#         if not isinstance(output["bear_case"], list):
#             output["bear_case"] = []
#         if not isinstance(output["key_risks"], list):
#             output["key_risks"] = []
#         if not isinstance(output["sources"], list):
#             output["sources"] = []

#         if len(output["sources"]) < 2:
#             grounded_sources = self._build_grounded_sources(
#                 rag_chunks=rag_chunks,
#                 web_results=web_results,
#                 report_files=report_files,
#             )
#             output["sources"] = self._dedupe_sources(output["sources"] + grounded_sources)

#         return output

#     def _build_grounded_sources(
#         self,
#         rag_chunks: List[Dict[str, Any]],
#         web_results: List[Dict[str, Any]],
#         report_files: List[Path],
#     ) -> List[Dict[str, Any]]:
#         sources: List[Dict[str, Any]] = []

#         for chunk in rag_chunks[:3]:
#             doc_name = chunk.get("doc", "unknown_report")
#             text = chunk.get("text", "")
#             snippet = text[:300].strip()
#             sources.append(
#                 {
#                     "source_type": "pdf",
#                     "title": doc_name,
#                     "snippet": snippet,
#                     "url": None,
#                 }
#             )

#         for item in web_results[:2]:
#             title = item.get("title", "web_result")
#             snippet = item.get("content", "")[:300].strip()
#             url = item.get("url")
#             sources.append(
#                 {
#                     "source_type": "web",
#                     "title": title,
#                     "snippet": snippet,
#                     "url": url,
#                 }
#             )

#         if not sources and report_files:
#             for p in report_files[:2]:
#                 sources.append(
#                     {
#                         "source_type": "pdf",
#                         "title": p.name,
#                         "snippet": "Matched report file used for retrieval.",
#                         "url": None,
#                     }
#                 )

#         return sources

#     def _dedupe_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         seen = set()
#         deduped: List[Dict[str, Any]] = []

#         for s in sources:
#             key = (
#                 s.get("source_type"),
#                 s.get("title"),
#                 s.get("url"),
#             )
#             if key in seen:
#                 continue
#             seen.add(key)
#             deduped.append(
#                 {
#                     "source_type": s.get("source_type", "unknown"),
#                     "title": s.get("title", ""),
#                     "snippet": s.get("snippet", ""),
#                     "url": s.get("url"),
#                 }
#             )

#         return deduped