
# pyrefly: ignore [missing-import]
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Optional
from utils.text_extractor import extract_text, chunk_text
from utils.hf_embeddings import get_embeddings
from services.file_service import get_file_service
import uuid

class EmbeddingService:

    
    def __init__(self, chroma_base_path: str = "./chroma"):
        self.chroma_base_path = chroma_base_path
        os.makedirs(chroma_base_path, exist_ok=True)
    
    def get_collection_name(self, subject: str, unit: str) -> str:

        # ChromaDB collection names must be alphanumeric with underscores
        return f"{subject}_{unit}".replace(" ", "_").replace("-", "_").lower()
    
    def get_or_create_collection(self, subject: str, unit: str):

        collection_name = self.get_collection_name(subject, unit)
        
        # Create persistent client
        client = chromadb.PersistentClient(
            path=os.path.join(self.chroma_base_path, collection_name)
        )
        
        # Get embeddings function
        embeddings = get_embeddings()
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"subject": subject, "unit": unit}
        )
        
        return collection, embeddings
    
    def process_and_embed_documents(self, subject: str, unit: str) -> Dict:

        file_service = get_file_service()
        
        # Get all documents
        documents = file_service.get_all_documents(subject, unit)
        
        if not documents:
            print(f"[EMBED] No documents found for {subject}/{unit}")
            return {
                "status": "error",
                "message": "No documents found for this subject/unit"
            }
        
        # Get or create collection
        collection, embeddings = self.get_or_create_collection(subject, unit)
        
        # Clear existing data in collection before re-processing
        # This prevents duplicate data on retries
        existing = collection.get()
        if existing and existing['ids']:
            print(f"[EMBED] Clearing {len(existing['ids'])} existing chunks from collection")
            collection.delete(ids=existing['ids'])
        
        total_chunks = 0
        processed_files = []
        errors = []
        
        for doc_path in documents:
            try:
                print(f"[EMBED] Processing: {os.path.basename(doc_path)}")
                
                # Extract text
                text = extract_text(doc_path)
                
                if not text:
                    print(f"[EMBED] WARNING: No text extracted from {doc_path}")
                    errors.append(f"No text extracted from {os.path.basename(doc_path)}")
                    continue
                
                print(f"[EMBED] Extracted {len(text)} chars from {os.path.basename(doc_path)}")
                
                # Chunk text
                chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
                
                if not chunks:
                    print(f"[EMBED] WARNING: No chunks created from {doc_path}")
                    errors.append(f"No chunks created from {os.path.basename(doc_path)}")
                    continue
                
                print(f"[EMBED] Created {len(chunks)} chunks, generating embeddings...")
                
                # Generate embeddings in batches to avoid memory issues
                batch_size = 50
                all_ids = []
                all_embeddings = []
                all_metadatas = []
                all_chunks = []
                
                for batch_start in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[batch_start:batch_start + batch_size]
                    batch_embeddings = embeddings.embed_documents(batch_chunks)
                    batch_ids = [str(uuid.uuid4()) for _ in batch_chunks]
                    batch_metadatas = [
                        {
                            "source": os.path.basename(doc_path),
                            "subject": subject,
                            "unit": unit,
                            "chunk_index": batch_start + i
                        }
                        for i in range(len(batch_chunks))
                    ]
                    
                    all_ids.extend(batch_ids)
                    all_embeddings.extend(batch_embeddings)
                    all_metadatas.extend(batch_metadatas)
                    all_chunks.extend(batch_chunks)
                    
                    print(f"[EMBED] Embedded batch {batch_start // batch_size + 1} ({len(batch_chunks)} chunks)")
                
                # Add all to collection
                collection.add(
                    ids=all_ids,
                    embeddings=all_embeddings,
                    documents=all_chunks,
                    metadatas=all_metadatas
                )
                
                total_chunks += len(all_chunks)
                processed_files.append({
                    "file": os.path.basename(doc_path),
                    "chunks": len(all_chunks)
                })
                
                print(f"[EMBED] Successfully embedded {len(all_chunks)} chunks from {os.path.basename(doc_path)}")
                
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(doc_path)}: {str(e)}"
                print(f"[EMBED] {error_msg}")
                print(f"[EMBED] Traceback: {__import__('traceback').format_exc()}")
                errors.append(error_msg)
                continue
        
        # Only mark embedding as done if we actually embedded some content
        if total_chunks > 0:
            file_service.mark_embedding_done(subject, unit)
            print(f"[EMBED] SUCCESS: Embedded {total_chunks} total chunks from {len(processed_files)} files for {subject}/{unit}")
            
            return {
                "status": "success",
                "subject": subject,
                "unit": unit,
                "total_chunks": total_chunks,
                "processed_files": processed_files,
                "collection_name": self.get_collection_name(subject, unit)
            }
        else:
            # Don't mark as done — embeddings failed
            print(f"[EMBED] FAILED: No chunks were embedded for {subject}/{unit}. Errors: {errors}")
            return {
                "status": "error",
                "subject": subject,
                "unit": unit,
                "message": f"Failed to embed documents. Errors: {'; '.join(errors) if errors else 'Unknown error'}",
                "total_chunks": 0
            }
    
    def query_documents(self, subject: str, unit: str, query: str, n_results: int = 5) -> List[Dict]:

        collection, embeddings = self.get_or_create_collection(subject, unit)
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results and results.get('documents') and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                    "distance": results['distances'][0][i] if results.get('distances') else None
                })
        
        # Fallback: if ChromaDB is empty or returns no results, try to read file content directly
        if not formatted_results:
            print(f"[EMBED] ChromaDB empty/no results for {subject}/{unit}, falling back to direct file reading for query")
            file_service = get_file_service()
            documents = file_service.get_all_documents(subject, unit)
            
            if documents:
                for doc_path in documents:
                    try:
                        text = extract_text(doc_path)
                        if text:
                            # Append the document text as a chunk (limiting length so it doesn't blow up the prompt context)
                            formatted_results.append({
                                "content": text[:12000],
                                "metadata": {"source": os.path.basename(doc_path)},
                                "distance": 0.0
                            })
                            print(f"[EMBED] Fallback query: Read {len(text)} chars from {os.path.basename(doc_path)}")
                    except Exception as e:
                        print(f"[EMBED] Fallback query: Error reading {doc_path}: {e}")
        
        return formatted_results[:n_results]
    
    def get_all_documents_content(self, subject: str, unit: str) -> str:

        collection, _ = self.get_or_create_collection(subject, unit)
        
        # Get all documents from collection
        results = collection.get()
        
        if results and results['documents'] and len(results['documents']) > 0:
            # Concatenate all chunks
            return "\n\n".join(results['documents'])
        
        # Fallback: If ChromaDB is empty, try reading directly from files on disk
        print(f"[EMBED] ChromaDB empty for {subject}/{unit}, falling back to direct file reading")
        file_service = get_file_service()
        documents = file_service.get_all_documents(subject, unit)
        
        if documents:
            all_text = []
            for doc_path in documents:
                try:
                    text = extract_text(doc_path)
                    if text:
                        all_text.append(text)
                        print(f"[EMBED] Fallback: Read {len(text)} chars from {os.path.basename(doc_path)}")
                except Exception as e:
                    print(f"[EMBED] Fallback: Error reading {doc_path}: {e}")
                    continue
            
            if all_text:
                return "\n\n".join(all_text)
        
        return ""

# Global instance
_embedding_service_instance = None

def get_embedding_service() -> EmbeddingService:

    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
