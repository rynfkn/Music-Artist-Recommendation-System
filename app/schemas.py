from pydantic import BaseModel
from typing import List, Optional

class ArtistRecommendation(BaseModel):
    artist_id: int
    artist_name: Optional[str]
    # artist_url: Optional[str]
    score: Optional[float]
    tag_count: Optional[int]

class RecommendationResponse(BaseModel):
    user_id: int
    top_k: int
    items: List[ArtistRecommendation]
