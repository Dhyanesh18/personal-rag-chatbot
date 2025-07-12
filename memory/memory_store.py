import time
import chromadb
from datetime import datetime
from typing import List, Dict, Any

class MemoryStore:
    def __init__(self, persist_dir="./chroma_store", collection_name="session_summaries"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store(self, summary: str, embedding, session_data : Dict[str, Any]):
        """Store a summarized memory or factual preference."""
        metadata = {
            "session_id": session_data.get("session_id", "unknown"),
            "timestamp": session_data.get("end_time", datetime.now().isoformat()),
            "message_count": session_data.get("message_count", 0)
        }
        session_id = session_data.get("session_id", int(time.time() * 1000000))
        doc_id = f"summary_{session_id}"
        self.collection.add(
            documents = [summary],
            embeddings = [embedding],
            ids = [doc_id],
            metadatas = [metadata]
        )

    def retrieve(self, query_embedding, top_k=10):
        """Retrieve session summaries based on query embedding"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )    
        summaries = []
        if results["documents"] and results["documents"][0]:
            for i, summary in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                distance = results["distances"][0][i] if results["distances"][0] else 1.0

                summaries.append({
                    "content": summary.strip(),
                    "relevance_score": 1.0 - distance,
                    "session_id": metadata.get("session_id", "unknown"),
                    "timestamp": metadata.get("timestamp", ""),
                    "message_count": metadata.get("message_count", 0)
                })
        summaries.sort(key=lambda x: x["relevance_score"], reverse=True)
        return summaries

    def retrieve_recent_summaries(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most recent session summaries."""
        try:
            results = self.collection.get(
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                return []
            
            # Combine documents with minimal metadata and sort by timestamp
            summaries = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                summaries.append({
                    "content": doc,
                    "session_id": metadata.get("session_id", "unknown"),
                    "timestamp": metadata.get("timestamp", ""),
                    "message_count": metadata.get("message_count", 0)
                })
            
            # Sort by timestamp (most recent first)
            summaries.sort(key=lambda x: x["timestamp"], reverse=True)
            return summaries[:limit]
            
        except Exception as e:
            print(f"Error retrieving recent summaries: {e}")
            return []

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about stored session summaries."""
        try:
            all_data = self.collection.get(include=["metadatas"])
            
            total_summaries = len(all_data["metadatas"]) if all_data["metadatas"] else 0
            
            return {
                "total_summaries": total_summaries
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_summaries": 0}

    def reset(self):
        """Reset the memory store by deleting and recreating the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print("Session summary memory store reset successfully")
        except Exception as e:
            print(f"Error resetting memory store: {e}")

    def cleanup_old_summaries(self, keep_recent: int = 50):
        """Keep only recent session summaries."""
        try:
            # Get all summaries
            results = self.collection.get(
                include=["metadatas", "ids"]
            )
            
            if not results["ids"] or len(results["ids"]) <= keep_recent:
                return
            
            # Sort by timestamp and get IDs of old summaries to delete
            summaries = list(zip(results["ids"], results["metadatas"]))
            summaries.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)
            
            # Get IDs of summaries to delete (keep only recent ones)
            old_summaries = summaries[keep_recent:]
            ids_to_delete = [summary_id for summary_id, _ in old_summaries]
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                print(f"Cleaned up {len(ids_to_delete)} old session summaries")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
