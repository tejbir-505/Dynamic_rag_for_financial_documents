import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
import requests
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

load_dotenv()

import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class RetrievedChunk:
    """Data class for retrieved chunks with scores"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: float
    bm25_score: float
    hybrid_score: float


class HybridRetriever:
    """
    Hybrid retriever that combines semantic similarity and keyword search (BM25)
    """
    
    def __init__(
        self,
        collection_name: str = "document_chunks",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "jina-embeddings-v2-base-en",
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize the hybrid retriever
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory where vector database is stored
            embedding_model: Jina AI embedding model name
            semantic_weight: Weight for semantic similarity scores
            bm25_weight: Weight for BM25 scores
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        
        # Validate weights
        if abs(semantic_weight + bm25_weight - 1.0) > 1e-6:
            raise ValueError("Semantic weight and BM25 weight must sum to 1.0")
        
        self.jina_api_key = os.getenv("JINA_API_KEY")
        if not self.jina_api_key:
            raise ValueError("JINA_API_KEY not found in environment variables")
        
        self.embedding_url = "https://api.jina.ai/v1/embeddings"
        
        # Initialize text processing tools
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to collection: {collection_name}")
        except:
            raise ValueError(f"Collection {collection_name} not found. Please ensure documents are embedded first.")
        
        # Initialize BM25 index
        self.bm25_index = None
        self.documents_cache = None
        self._build_bm25_index()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing
        
        Args:
            text: Input text
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return processed_tokens
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection"""
        logger.info("Building BM25 index...")
        
        try:
            # Get all documents from Chroma (ids are returned by default)
            results = self.collection.get(include=["documents", "metadatas"])
            
            if not results["documents"]:
                logger.warning("No documents found in collection")
                return
            
            # Cache documents with metadata
            self.documents_cache = [
                {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                for i in range(len(results["documents"]))
            ]
            
            # Preprocess documents for BM25
            processed_docs = [
                self._preprocess_text(doc["content"]) 
                for doc in self.documents_cache
            ]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(processed_docs)
            
            logger.info(f"BM25 index built with {len(processed_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for the query using Jina AI
        
        Args:
            query: Search query
            
        Returns:
            Query embedding vector
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}"
        }
        
        payload = {
            "model": self.embedding_model,
            "input": [query],
        }
        
        try:
            response = requests.post(
                self.embedding_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    def _semantic_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Perform semantic search using Chroma
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of semantic search results
        """
        try:
            query_embedding = self._get_query_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            semantic_results = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = 1 - distance  
                
                semantic_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "semantic_score": max(0, similarity)  
                })
            
            return semantic_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise
    
    def _bm25_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of BM25 search results
        """
        if self.bm25_index is None or self.documents_cache is None:
            logger.warning("BM25 index not available")
            return []
        
        try:
            processed_query = self._preprocess_text(query)
            
            if not processed_query:
                logger.warning("Query preprocessing resulted in empty tokens")
                return []
            
            scores = self.bm25_index.get_scores(processed_query)
            
            # Get top N results
            top_indices = np.argsort(scores)[::-1][:n_results] # sort the scores in descending order and get top n_results
            
            bm25_results = []
            for idx in top_indices:
                if scores[idx] > 0: 
                    bm25_results.append({
                        "id": self.documents_cache[idx]["id"],
                        "content": self.documents_cache[idx]["content"],
                        "metadata": self.documents_cache[idx]["metadata"],
                        "bm25_score": float(scores[idx])
                    })
            
            return bm25_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _combine_results(
        self, 
        semantic_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]]
    ) -> List[RetrievedChunk]:
        """
        Combine and score results from both search methods
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            
        Returns:
            List of RetrievedChunk objects with hybrid scores
        """
        # Create a mapping for lookup
        semantic_map = {result["id"]: result for result in semantic_results}
        bm25_map = {result["id"]: result for result in bm25_results}
        
        # Get all unique document ids
        all_ids = set(semantic_map.keys()) | set(bm25_map.keys())
        
        # Normalize scores
        if semantic_results:
            max_semantic = max(r["semantic_score"] for r in semantic_results)
            min_semantic = min(r["semantic_score"] for r in semantic_results)
            semantic_range = max_semantic - min_semantic if max_semantic != min_semantic else 1
        else:
            max_semantic = min_semantic = semantic_range = 1
        
        if bm25_results:
            max_bm25 = max(r["bm25_score"] for r in bm25_results)
            min_bm25 = min(r["bm25_score"] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
        else:
            max_bm25 = min_bm25 = bm25_range = 1
        
        combined_results = []
        
        for doc_id in all_ids:
            # Get scores (0 if not found in that search method)
            semantic_score = 0
            bm25_score = 0
            content = ""
            metadata = {}
            
            if doc_id in semantic_map:
                semantic_result = semantic_map[doc_id]
                semantic_score = semantic_result["semantic_score"]
                content = semantic_result["content"]
                metadata = semantic_result["metadata"]
            
            if doc_id in bm25_map:
                bm25_result = bm25_map[doc_id]
                bm25_score = bm25_result["bm25_score"]
                if not content:  # If not from semantic search
                    content = bm25_result["content"]
                    metadata = bm25_result["metadata"]
            
            # Normalize scores to [0, 1]
            normalized_semantic = (semantic_score - min_semantic) / semantic_range if semantic_range > 0 else 0
            normalized_bm25 = (bm25_score - min_bm25) / bm25_range if bm25_range > 0 else 0
            
            # Calculate hybrid score
            hybrid_score = (
                self.semantic_weight * normalized_semantic + 
                self.bm25_weight * normalized_bm25
            )
            
            retrieved_chunk = RetrievedChunk(
                chunk_id=doc_id,
                content=content,
                metadata=metadata,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                hybrid_score=hybrid_score
            )
            
            combined_results.append(retrieved_chunk)
        
        # Sort by hybrid score (descending)
        combined_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return combined_results
    
    def retrieve(
        self, 
        query: str, 
        n_results: int = 5,
        semantic_n: Optional[int] = None,
        bm25_n: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve documents using hybrid search
        
        Args:
            query: Search query
            n_results: Number of final results to return
            semantic_n: Number of results from semantic search (default: 2*n_results)
            bm25_n: Number of results from BM25 search (default: 2*n_results)
            
        Returns:
            List of RetrievedChunk objects ranked by hybrid score
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Default to getting more results from each method for better fusion
        if semantic_n is None:
            semantic_n = min(2 * n_results, 20)
        if bm25_n is None:
            bm25_n = min(2 * n_results, 20)
        
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        
        try:
            # Perform both searches
            semantic_results = self._semantic_search(query, semantic_n)
            bm25_results = self._bm25_search(query, bm25_n)
            
            logger.info(f"Semantic search returned {len(semantic_results)} results")
            logger.info(f"BM25 search returned {len(bm25_results)} results")
            
            # Combine results
            combined_results = self._combine_results(semantic_results, bm25_results)
            
            # Return top N results
            final_results = combined_results[:n_results]
            
            logger.info(f"✅ Returning {len(final_results)} hybrid search results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            raise
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever
        
        Returns:
            Dictionary with retriever statistics
        """
        try:
            collection_count = self.collection.count()
            bm25_docs = len(self.documents_cache) if self.documents_cache else 0
            
            return {
                "collection_name": self.collection_name,
                "total_documents": collection_count,
                "bm25_indexed_documents": bm25_docs,
                "semantic_weight": self.semantic_weight,
                "bm25_weight": self.bm25_weight,
                "bm25_index_ready": self.bm25_index is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting retriever stats: {e}")
            return {"error": str(e)}


# Example usage and testing functions
def example_usage():
    """Example of how to use the hybrid retriever"""
    
    retriever = HybridRetriever(
        collection_name="document_chunks",
        persist_directory="./chroma_db",
        semantic_weight=0.7,
        bm25_weight=0.3
    )
    
    # Example queries
    queries = [
        "What are the executive officers?",
        "financial performance revenue growth",
        "risk factors and challenges",
        "balance sheet assets liabilities"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        results = retriever.retrieve(query, n_results=3)
        
        for i, chunk in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Section: {chunk.metadata.get('section_title', 'Unknown')}")
            print(f"Semantic Score: {chunk.semantic_score:.3f}")
            print(f"BM25 Score: {chunk.bm25_score:.3f}")
            print(f"Hybrid Score: {chunk.hybrid_score:.3f}")
            print(f"Content Preview: {chunk.content[:100]}...")
    
    # Print stats
    stats = retriever.get_retriever_stats()
    print(f"\nRetriever Stats: {stats}")


if __name__ == "__main__":
    example_usage()