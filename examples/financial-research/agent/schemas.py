from pydantic import BaseModel
from typing import List, Optional


class Source(BaseModel):
    source_type: str
    title: str
    snippet: str
    page: Optional[int] = None
    url: Optional[str] = None


class FinancialAgentOutput(BaseModel):
    answer: str
    evidence: List[Source]
    uncertainty: Optional[str] = None



# from pydantic import BaseModel, Field
# from typing import List, Optional


# class Source(BaseModel):
#     source_type: str  # "pdf" | "financial_dataset" | "web"
#     title: str
#     snippet: str
#     url: Optional[str] = None


# class FinancialAgentInput(BaseModel):
#     role: str
#     company: str
#     ticker: str
#     goal: str
#     constraints: List[str] = Field(default_factory=list)


# class FinancialAgentOutput(BaseModel):
#     summary: str
#     bull_case: List[str]
#     bear_case: List[str]
#     key_risks: List[str]
#     conclusion: str
#     confidence: str
#     sources: List[Source]