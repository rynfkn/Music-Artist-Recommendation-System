from .config import NEO4J_DATABASE

def get_user_top_tags(driver, user_id, limit=20):
    query = """
    MATCH (u:User {userID: $user_id})-[:LISTENED]->(:Artist)-[:HAS_TAG]->(t:Tag)
    RETURN t.value AS tag, count(*) AS freq
    ORDER BY freq DESC
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        res = session.run(query, user_id=user_id, limit=limit)
        return [(r["tag"], r["freq"]) for r in res]

def get_artist_tags(driver, artist_id):
    query = """
    MATCH (a:Artist {id: $artist_id})-[:HAS_TAG]->(t:Tag)
    RETURN t.value AS tag
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        res = session.run(query, artist_id=artist_id)
        return [r["tag"] for r in res]

def get_friends_who_listened(driver, user_id, artist_id):
    query = """
    MATCH (u:User {userID: $user_id})-[:FRIEND]->(f:User)-[:LISTENED]->(a:Artist {id: $artist_id})
    RETURN f.userID AS friend_id
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        res = session.run(query, user_id=user_id, artist_id=artist_id)
        return [r["friend_id"] for r in res]


def get_similar_artists_by_tag(driver, user_id, artist_id, limit=5):
    query = """
    MATCH (a2:Artist {id: $artist_id})-[:HAS_TAG]->(t:Tag)
    MATCH (u:User {userID: $user_id})-[:LISTENED]->(a1:Artist)-[:HAS_TAG]->(t)
    RETURN a1.id AS artist_id, a1.name AS name, count(t) AS shared_tags
    ORDER BY shared_tags DESC
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        res = session.run(query, user_id=user_id, artist_id=artist_id, limit=limit)
        return [{"artist_id": r["artist_id"], "artist_name": r["name"], "shared_tags": r["shared_tags"]} for r in res]


def get_embedding_similarity(driver, artist_id, limit=5):
    query = """
    MATCH (a2:Artist {id: $artist_id})
    MATCH (a1:Artist)
    WHERE a1.embedding IS NOT NULL AND a2.embedding IS NOT NULL
    WITH a1, gds.similarity.cosine(a1.embedding, a2.embedding) AS sim
    ORDER BY sim DESC
    LIMIT $limit
    RETURN a1.id AS artist_id, a1.name AS name, sim
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        res = session.run(query, artist_id=artist_id, limit=limit)
        return [{"artist_id": r["artist_id"], "artist_name": r["name"], "score": float(r["sim"])} for r in res]
