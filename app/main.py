from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .recommender import RecommenderService
from .schemas import RecommendationResponse, ArtistRecommendation
from .config import DEFAULT_TOP_K
from .explainer_service import get_user_top_tags, get_artist_tags, get_friends_who_listened, get_similar_artists_by_tag, get_embedding_similarity

app = FastAPI(
    title="Music Artist Recommendation API",
    version="1.0.0"
)

origins = [
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender_service: RecommenderService | None = None

@app.on_event("startup")
def startup_event():
    global recommender_service
    recommender_service = RecommenderService()
    print("[API] RecommenderService initialized.")

@app.on_event("shutdown")
def shutdown_event():
    global recommender_service
    if recommender_service is not None:
        recommender_service.close()
        print("[API] RecommenderService closed.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse
)
def get_recommendations(user_id: int, k: int = DEFAULT_TOP_K):
    global recommender_service
    if recommender_service is None:
        raise HTTPException(status_code=500, detail="Recommender not initialized")

    recs = recommender_service.recommend_for_user(user_id=user_id, top_k=k)
    print(f">>>>> {recs}")

    items = [ArtistRecommendation(**r) for r in recs]

    return RecommendationResponse(
        user_id=user_id,
        top_k=k,
        items=items
    )


@app.get("/explain", response_model=dict)
def explain_recommendation(user_id: int, artist_id: int):
    global recommender_service
    driver = recommender_service.driver
    
    user_tags = get_user_top_tags(driver, user_id)
    artist_tags = get_artist_tags(driver, artist_id)
    shared_tags = list(set(t for t,_ in user_tags) & set(artist_tags))
    
    friends = get_friends_who_listened(driver, user_id, artist_id)

    similar_artists = get_similar_artists_by_tag(driver, user_id, artist_id)

    embedding_sims = get_embedding_similarity(driver, artist_id)

    return {
        "user_id": user_id,
        "artist_id": artist_id,
        "explanations": {
            "tag_overlap": {
                "user_tags": [t for t,_ in user_tags],
                "artist_tags": artist_tags,
                "shared": shared_tags
            },
            "friend_activity": {
                "friends_who_listened": friends
            },
            "similar_artists": similar_artists,
            "embedding_similarity": embedding_sims
        }
    }
