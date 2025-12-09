import numpy as np
from neo4j import GraphDatabase
import joblib
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MODEL_PATH, DEFAULT_TOP_K

class RecommenderService:
    def __init__(self):
        # connect ke Neo4j
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # load model
        self.model = joblib.load(MODEL_PATH)

        # cache artist embeddings di memory (sangat disarankan)
        self.artist_embeddings = self._load_artist_embeddings()

    def close(self):
        self.driver.close()


    def _load_artist_embeddings(self):
        """
        Load semua embedding artist sekali di startup,
        simpan di dict {artist_id: np.array(embedding)}.
        """
        query = """
        MATCH (a:Artist)
        WHERE a.embedding IS NOT NULL
        RETURN a.id AS artist_id, a.embedding AS emb
        """

        artist_embs = {}
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                artist_id = record["artist_id"]
                emb = np.array(record["emb"], dtype=float)
                artist_embs[artist_id] = emb

        print(f"[Recommender] Loaded {len(artist_embs)} artist embeddings.")
        return artist_embs

    def _get_user_embedding(self, user_id: int):
        """
        Ambil embedding user dari Neo4j.
        """
        query = """
        MATCH (u:User {userID: $user_id})
        WHERE u.embedding IS NOT NULL
        RETURN u.embedding AS emb
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id)
            record = result.single()

        if record is None:
            return None

        emb = np.array(record["emb"], dtype=float)
        return emb

    def _get_existing_links(self, user_id: int):
        """
        Ambil artist yang sudah pernah didengarkan user ini
        supaya tidak direkomendasikan lagi.
        """
        query = """
        MATCH (u:User {userID: $user_id})-[:LISTENED]->(a:Artist)
        RETURN a.id AS artist_id
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id)
            return {r["artist_id"] for r in result}

    def _get_artist_meta_bulk(self, artist_ids):
        """
        Ambil metadata artist (name, url, dsb) sekaligus.
        """
        if not artist_ids:
            return {}

        query = """
        UNWIND $ids AS aid
        MATCH (a:Artist {id: aid})
        RETURN a.id AS artist_id, a.name AS name, a.url AS url
        """
        with self.driver.session() as session:
            result = session.run(query, ids=list(artist_ids))
            meta = {}
            for r in result:
                meta[r["artist_id"]] = {
                    "name": r["name"],
                    # "url": r["url"]
                }
        return meta

    def recommend_for_user(self, user_id: int, top_k: int = None):
        """
        Kembalikan top_k rekomendasi artist untuk user_id tertentu
        beserta skor model.
        """
        if top_k is None:
            top_k = DEFAULT_TOP_K

        # 1. ambil embedding user
        user_emb = self._get_user_embedding(user_id)
        if user_emb is None:
            # cold-start user: tidak punya embedding
            return self._recommend_for_cold_start_user(user_id, top_k)

        # 2. ambil artist yang sudah pernah didengar (supaya tidak direkomendasikan lagi)
        existing_links = self._get_existing_links(user_id)

        # 3. siapkan kandidat
        candidate_features = []
        candidate_artist_ids = []

        for artist_id, artist_emb in self.artist_embeddings.items():
            if artist_id in existing_links:
                continue

            # fitur pair: sama seperti di training (Hadamard product)
            feature_vector = user_emb * artist_emb
            candidate_features.append(feature_vector)
            candidate_artist_ids.append(artist_id)

        if not candidate_features:
            # semua artist sudah pernah didengar
            return []

        X_candidates = np.vstack(candidate_features)

        # 4. prediksi probability link LISTENED (class 1)
        probs = self.model.predict_proba(X_candidates)[:, 1]

        # 5. ambil Top-K
        top_k = min(top_k, len(candidate_artist_ids))
        top_indices = np.argsort(probs)[-top_k:][::-1]

        selected_ids = [candidate_artist_ids[i] for i in top_indices]
        selected_scores = [float(probs[i]) for i in top_indices]

        # 6. ambil meta info artist (name, url) sekaligus
        meta = self._get_artist_meta_bulk(selected_ids)

        recommendations = []
        for aid, score in zip(selected_ids, selected_scores):
            m = meta.get(aid, {})
            recommendations.append({
                "artist_id": aid,
                "artist_name": m.get("name"),
                # "artist_url": m.get("url"),
                # "score": score
            })

        return recommendations

    def _recommend_for_cold_start_user(self, user_id: int, top_k: int):
        """
        Fallback: user belum punya embedding (belum ada LISTENED).
        Bisa pakai popularity-based recommendation.
        """
        query = """
        MATCH (u:User)-[r:LISTENED]->(a:Artist)
        RETURN a.id AS artist_id, a.name AS name, a.url AS url, sum(r.weight) AS listen_score
        ORDER BY listen_score DESC
        LIMIT $top_k
        """
        with self.driver.session() as session:
            result = session.run(query, top_k=top_k)
            recs = []
            for r in result:
                recs.append({
                    "artist_id": r["artist_id"],
                    "artist_name": r["name"],
                    # "artist_url": r["url"],
                    # "score": float(r["listen_score"])  # di fallback ini skor = popularitas
                })
        
        return recs