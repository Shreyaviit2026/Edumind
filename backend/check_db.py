import sys
sys.path.insert(0, '.')
from services.embedding_service import get_embedding_service

embedding_service = get_embedding_service()
collection, _ = embedding_service.get_or_create_collection("Cloud", "Unit 1")

results = collection.get()
print(f"Number of documents in ChromaDB for Cloud/Unit 1: {len(results['documents'])}")
