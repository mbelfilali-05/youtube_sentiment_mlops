from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

MAX_BATCH = 500
MAX_CHARS = 5000


class BatchRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1, max_length=MAX_BATCH)

    @field_validator("comments")
    @classmethod
    def check_comments(cls, v):
        cleaned = [c[:MAX_CHARS] for c in v]
        if all(not c.strip() for c in cleaned):
            raise ValueError("All comments are empty")
        return cleaned


class Prediction(BaseModel):
    text: str
    label: int
    sentiment: str
    confidence: float


class BatchResponse(BaseModel):
    predictions: List[Prediction]
    total: int
    counts: Dict[str, int]
    percentages: Dict[str, float]
    inference_ms: float