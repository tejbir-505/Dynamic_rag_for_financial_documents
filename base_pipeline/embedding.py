import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
import requests
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from chunker import get_chunks

load_dotenv()
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Embedding manager that processes chunks and stores them in Chroma vector store
    using Jina embeding
    """
    
    def __init__(
        self, 
        collection_name: str = "document_chunks",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "jina-embeddings-v2-base-en"
    ):
        """
        Initialize the embedding manager
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the vector database
            embedding_model: Jina AI embedding model name
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Get Jina AI API key from environment
        self.jina_api_key = os.getenv("JINA_API_KEY")
        if not self.jina_api_key:
            raise ValueError("JINA_API_KEY not found in environment variables")
        
        # Jina AI API endpoint
        self.embedding_url = "https://api.jina.ai/v1/embeddings"
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Document chunks with semantic embeddings"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from Jina AI API with retry logic
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}"
        }
        
        payload = {
            "model": self.embedding_model,
            "input": texts,
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
            embeddings = [item["embedding"] for item in result["data"]]
            
            logger.info(f"Successfully embedded {len(texts)} texts")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise
    
    def prepare_chunk_content(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare chunk content
        
        Args:
            chunk: Chunk dictionary from chunker module
            
        Returns:
            Formatted content string
        """
        section_title = chunk.get("section_title", "Unknown Section")
        full_content = chunk.get("full_content", "")
        
        # Strategy 1 format: Section title + content
        enhanced_content = f"Section: {section_title}\n\nContent: {full_content}"
        
        return enhanced_content

    def _process_single_batch(self, batch: List[Dict[str, Any]], batch_start_idx: int) -> None:
        """
        Process a single batch of chunks
        
        Args:
            batch: Batch of chunks to process
            batch_start_idx: Starting index for logging
        """
        try:
            # Prepare content for embedding
            texts_to_embed = []
            chunk_metadata = []
            chunk_ids = []
            
            for chunk in batch:
                # Prepare enhanced content
                enhanced_content = self.prepare_chunk_content(chunk)
                texts_to_embed.append(enhanced_content)
                
                # Prepare metadata (only essential fields)
                metadata = {
                    "chunk_id": chunk.get("chunk_id", f"chunk_{uuid.uuid4()}"),
                    "section_title": chunk.get("section_title", "Unknown"),
                    "page_numbers": str(chunk.get("page_numbers", [1])),
                    "document_title": chunk.get("document_title", "Unknown Document"),
                    "content_length": len(enhanced_content)
                }
                chunk_metadata.append(metadata)
                
                # Use chunk_id as the unique identifier for Chroma
                chunk_ids.append(chunk["chunk_id"])
            

            embeddings = self.get_embeddings(texts_to_embed)
            
            # Add to Chroma collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts_to_embed,
                metadatas=chunk_metadata,
                ids=chunk_ids
            )
            
            logger.info(f"Successfully processed batch {batch_start_idx//16 + 1} ({len(batch)} chunks)")
            
        except Exception as e:
            logger.error(f"Error processing batch starting at index {batch_start_idx}: {e}")
            raise
    
    def process_chunks_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 16) -> None:
        """
        Process chunks in batches and store in vector database
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks to process at once
        """
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._process_single_batch(batch, i)
            
            # Small delay to respect API rate limits
            time.sleep(0.1)
    

    
    def embed_pdf_document(self, pdf_path: str, document_title: str = None, max_chunk_size: int = 800) -> int:
        """
        Process a PDF document and store embeddings
        
        Args:
            pdf_path: Path to PDF file
            document_title: Title of the document
            max_chunk_size: Maximum chunk size for processing
            
        Returns:
            Number of chunks processed
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Get chunks from your chunker module
            chunks_dict, langchain_docs = get_chunks(
                pdf_path=pdf_path,
                document_title=document_title,
                max_chunk_size=max_chunk_size
            )
            for i, chunk in enumerate(chunks_dict[:10]):
                print(f"Chunk {i}: ID={chunk.get('chunk_id')}, Section={chunk.get('section_title')}")
            
            if not chunks_dict:
                logger.warning("No chunks received from chunker")
                return 0
            
            # Process chunks in batches
            # self.process_chunks_batch(chunks_dict)
            
            logger.info(f"Successfully embedded {len(chunks_dict)} chunks from {pdf_path}")
            return len(chunks_dict)
            
        except Exception as e:
            logger.error(f"Error embedding PDF {pdf_path}: {e}")
            raise
    
    def search_similar_chunks(
        self, 
        query: str, 
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search in Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                result = {
                    "chunk_id": results["metadatas"][0][i]["chunk_id"],
                    "content": results["documents"][0][i],
                    "section_title": results["metadatas"][0][i]["section_title"],
                    "page_numbers": results["metadatas"][0][i]["page_numbers"],
                    "document_title": results["metadatas"][0][i]["document_title"],
                    "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "distance": results["distances"][0][i]
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar chunks for query: '{query[:50]}...'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(limit=min(100, count), include=["metadatas"])
            
            stats = {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
            if sample_results["metadatas"]:
                # Analyze sample metadata
                documents = set()
                sections = set()
                
                for metadata in sample_results["metadatas"]:
                    documents.add(metadata.get("document_title", "Unknown"))
                    sections.add(metadata.get("section_title", "Unknown"))
                
                stats.update({
                    "unique_documents": len(documents),
                    "unique_sections": len(sections),
                    "sample_documents": list(documents)[:5],
                    "sample_sections": list(sections)[:5]
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection (use with caution!)
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False


# Example usage and testing functions
def example_usage():
    """Example of how to use the embedding manager"""
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(
        collection_name="my_documents",
        persist_directory="./my_vector_db"
    )
    
    # Process a PDF document
    pdf_path = "C:/Users/tejup/Downloads/extraction_purpose2.pdf"
    num_chunks = embedding_manager.embed_pdf_document(
        pdf_path=pdf_path,
        document_title="My Important Document"
    )
    
    print(f"Processed {num_chunks} chunks")
    
    # Search for similar content
    query = "who was president and chied executive officer in 2012"
    results = embedding_manager.search_similar_chunks(
        query=query,
        n_results=3
    )
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Section: {result['section_title']}")
        print(f"Pages: {result['page_numbers']}")
        print(f"Similarity: {result['similarity_score']:.3f}")
        print(f"Content preview: {result['content'][:200]}...")
    
    # Get collection statistics
    stats = embedding_manager.get_collection_stats()
    print(f"\nCollection Stats: {stats}")


if __name__ == "__main__":
    example_usage()