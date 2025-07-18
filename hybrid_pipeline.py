"""
hybrid_pipeline.py

Hybrid Retrieval Pipeline for combining sparse (BM25) and dense (embedding-based) search
with Reciprocal Rank Fusion (RRF).

This module provides a robust hybrid retrieval system using:
    - Elasticsearch for BM25-based lexical search
    - ChromaDB for vector-based dense retrieval
    - A custom embedder for generating embeddings
    - Reciprocal Rank Fusion to combine heterogeneous results
"""

from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from elasticsearch.helpers import bulk
from typing import List, Dict, Any


class HybridRetrievalPipeline:
    """
    HybridRetrievalPipeline

    Combines BM25 (Elasticsearch) and dense vector search (ChromaDB)
    using Reciprocal Rank Fusion for robust hybrid retrieval.
    """

    def __init__(self, es_client: Elasticsearch, top_k: int = 50) -> None:
        """
        Initialize the Hybrid Retrieval Pipeline.

        Args:
            es_client (Elasticsearch): An initialized Elasticsearch client.
            top_k (int, optional): Number of candidates to retrieve for each retriever. Defaults to 50.
        """
        self.embedder = Embedder()
        self.memory = MemoryStore(collection_name="documents")
        self.es = es_client
        self.top_k = top_k

        self._ensure_index()

    def _ensure_index(self) -> None:
        """
        Ensure that the Elasticsearch index for summaries exists.
        If it does not exist, create it with appropriate mappings.
        """
        index_name = "summaries"
        print(f"Checking if index '{index_name}' exists...")

        try:
            exists = self.es.indices.exists(index=index_name)
            print(f"indices.exists() returned: {exists}")
        except es_exceptions.ApiError as e:
            print(f"Elasticsearch API error: {repr(e)}")
            raise
        except Exception as e:
            print(f"Unexpected exception: {repr(e)}")
            raise

        if not exists:
            print(f"Index '{index_name}' does not exist. Creating...")
            self.es.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )
            print(f"Created index '{index_name}'.")
        else:
            print(f"Index '{index_name}' already exists.")

    def rebuild_bm25_index(self) -> None:
        """
        Rebuild the entire BM25 index in Elasticsearch.

        This will:
            - Delete the existing index (if it exists)
            - Recreate it with the defined mappings
            - Reindex all documents from the dense vector store (ChromaDB)
        """
        index_name = "summaries"
        print("Rebuilding BM25 index in Elasticsearch...")

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            print(f"Deleted existing index '{index_name}'.")

        self._ensure_index()

        results = self.memory.collection.get()
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            print("No documents found in ChromaDB for indexing.")
            return

        actions = [
            {
                "_index": index_name,
                "_source": {"text": doc, "metadata": meta}
            }
            for doc, meta in zip(documents, metadatas)
        ]

        success, _ = bulk(self.es, actions)
        print(f"Successfully indexed {success} documents into Elasticsearch.")

    def _bm25_search(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """
        Execute a BM25 search in Elasticsearch.

        Args:
            query (str): The input query string.
            top_k (int, optional): Number of results to return. Defaults to 25.

        Returns:
            List[Dict[str, Any]]: List of matched documents with scores.
        """
        res = self.es.search(
            index="summaries",
            body={
                "query": {
                    "match": {
                        "text": {
                            "query": query
                        }
                    }
                },
                "size": top_k
            }
        )

        return [
            {
                "text": hit["_source"]["text"],
                "meta": hit["_source"]["metadata"],
                "score": hit["_score"],
                "retrieval_method": "bm25"
            }
            for hit in res["hits"]["hits"]
        ]

    def _dense_search(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """
        Execute a dense vector search using the Embedder + ChromaDB.

        Args:
            query (str): The input query string.
            top_k (int, optional): Number of results to return. Defaults to 25.

        Returns:
            List[Dict[str, Any]]: List of matched documents with similarity scores.
        """
        query_embedding = self.embedder.get_embedding(query)
        raw_results = self.memory.retrieve(query_embedding=query_embedding, top_k=top_k)

        return [
            {
                "text": chunk["text"],
                "meta": chunk["metadata"],
                "score": 1.0 / (1.0 + chunk["score"]),
                "retrieval_method": "dense"
            }
            for chunk in raw_results
        ]

    def reciprocal_rank_fusion(
        self, result_lists: List[List[Dict[str, Any]]], k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion (RRF).

        Args:
            result_lists (List[List[Dict[str, Any]]]): Lists of ranked results to fuse.
            k (int, optional): RRF constant. Defaults to 60.

        Returns:
            List[Dict[str, Any]]: Fused and re-ranked list.
        """
        fused_scores = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = hash(doc["text"])
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0,
                        "methods": []
                    }
                fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)
                fused_scores[doc_id]["methods"].append(doc["retrieval_method"])

        fused = [
            {
                **item["doc"],
                "rrf_score": item["score"],
                "retrieval_methods": list(set(item["methods"])),
                "score": item["score"],
            }
            for item in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        ]

        return fused

    def retrieve(self, query: str, final_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid BM25 + Dense + RRF.

        Args:
            query (str): User query.
            final_k (int, optional): Number of final results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: Ranked list of relevant documents.
        """
        dense_results = self._dense_search(query, top_k=self.top_k // 2)
        bm25_results = self._bm25_search(query, top_k=self.top_k // 2)

        print(f"Dense results retrieved: {len(dense_results)}")
        print(f"BM25 results retrieved: {len(bm25_results)}")

        if dense_results and bm25_results:
            fused_results = self.reciprocal_rank_fusion([dense_results, bm25_results])
        elif dense_results:
            fused_results = dense_results
        elif bm25_results:
            fused_results = bm25_results
        else:
            return []

        print(f"Results after RRF fusion: {len(fused_results)}")

        return fused_results[:final_k]
