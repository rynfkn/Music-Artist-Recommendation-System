import os

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DB") or "fp-rsbp"
NEO4J_DB = NEO4J_DATABASE  # backwards compatibility for existing imports

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_model2.pkl")

DEFAULT_TOP_K = 10
