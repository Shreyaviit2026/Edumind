import sys
import os

# Add backend directory to sys.path
sys.path.append(os.path.abspath('.'))

from services.embedding_service import get_embedding_service

try:
    svc = get_embedding_service()
    print("Starting process_and_embed_documents...")
    result = svc.process_and_embed_documents("Cloud", "Unit 1")
    print(f"Result: {result}")
except Exception as e:
    import traceback
    traceback.print_exc()
