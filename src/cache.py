from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "fast_ingest"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"), 
    api_key=os.getenv("QDRANT_API_KEY"),)

# One Time
# qdrant.recreate_collection(
#     collection_name="fact_cache",
#     vectors_config=VectorParams(
#         size=512,
#         distance=Distance.COSINE
#     )
# )

def store_claim_cache(embedding, claim, credibility, content):
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": abs(hash(claim)),
            "vector": embedding,
            "payload": {
                "type": "claim",
                "modality": "text",
                "source": "PROJECT",
                "credibility": credibility,
                "content": content
            }
        }]
    )

def retrieve_similar_claims(embedding, top_k=3):
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=top_k,
        score_threshold=0.5
    ).points
