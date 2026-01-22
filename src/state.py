from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal


class FactCheckState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    input_type: Literal["text", "audio"]
    audio_path: str
    claim: str
    embedding: list[float]
    verdict: Literal["True", "False", "Mixed", "Unclear"]
    confidence: float
    evidence: str
    cache_hit: bool
    cached_results: list
    

class LLMResponse(BaseModel):
    verdict: Literal["True", "False", "Mixed", "Unclear"] = Field(..., description="The verdict on the claim.")
    confidence: float = Field(..., description="The confidence in the final verdict.")
    evidence: str = Field(..., description="The evidence supporting the verdict.")