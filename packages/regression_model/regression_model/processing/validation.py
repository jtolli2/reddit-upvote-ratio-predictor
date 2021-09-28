from regression_model.config import config
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Tuple

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    return input_data

class RedditInputSchema(BaseModel):
    title: Optional[str]
    selftext: Optional[str]
    subreddit_subscribers: Optional[int]
    spoiler: Optional[bool]
    over_18: Optional[bool]
    created_utc: Optional[int]
    is_original_content: Optional[bool]

class MultipleRedditInputs(BaseModel):
    inputs: List[RedditInputSchema]