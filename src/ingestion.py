import wikipedia
import json
import os
import uuid
from src.cache import qdrant, COLLECTION_NAME
from src.utils import embed_text, chunk_text

def download_wikipedia_topic(topic, save_dir="data/raw/wikipedia"):
    os.makedirs(save_dir, exist_ok=True)
    filename = topic.replace(" ", "_") + ".jsonl"
    path = os.path.join(save_dir, filename)

    if os.path.exists(path):
        print(f"[CACHE] Exists: {topic}")
        return path

    try:
        page = wikipedia.page(topic, auto_suggest=False)
        record = {"title": page.title, "url": page.url, "content": page.content}
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[SAVED] {topic}")
        return path
    except Exception as e:
        print(f"[ERROR] Could not download {topic}: {e}")
        return None

def ingest_wikipedia(topic):
    path = download_wikipedia_topic(topic)
    if not path: return

    with open(path, "r", encoding="utf-8") as f:
        page = json.loads(f.readline())

    print(f"[INGEST] Processing: {topic}")
    chunks = chunk_text(page['content'])

    points = []
    for chunk in chunks:
        vector = embed_text(chunk)
        payload = {
            "type": "evidence",
            "modality": "text",
            "source": "wikipedia",
            "credibility": 0.85,
            "topic": topic,
            "content": chunk
        }
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": payload
        })
    
    # Upsert in batch
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[COMPLETED] {topic}")