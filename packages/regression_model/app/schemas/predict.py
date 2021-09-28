from typing import Any, List, Optional
from pydantic import BaseModel
from regression_model.processing.validation import RedditInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[float]


class MultipleRedditInputs(BaseModel):
    inputs: List[RedditInputSchema]