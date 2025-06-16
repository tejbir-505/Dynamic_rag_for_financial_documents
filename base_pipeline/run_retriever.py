from typing import List
from embedding import EmbeddingManager
from retriever import RetrievedChunk, HybridRetriever  # Fixed: RetrievedChunk (not Retrievedchunk)

def embed(collection_name: str, 
    persist_directory: str, 
    pdf_path: str,
    document_title: str = "My important document", 
    queries: List[str] = None,
    n_results: int = 3,
    semantic_weight: float = 0.7, 
    bm25_weight: float = 0.3
):
    """Whole embedding pipeline"""
    print("STARTING RAG PIPELINE")
    
    # Step 1: Embed the document
    print("\n📄 STEP 1: EMBEDDING DOCUMENT")
    print("-" * 30)
    
    embedding_manager = EmbeddingManager(
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    num_chunks = embedding_manager.embed_pdf_document(
        pdf_path=pdf_path,
        document_title=document_title
    )

    print(f"Processed {num_chunks} chunks")

    # Get and display collection stats
    stats = embedding_manager.get_collection_stats()
    print(f"📊 Collection Stats: {stats}")


# def retrieve()

def run(
    collection_name: str, 
    persist_directory: str, 
    pdf_path: str,
    document_title: str = "My important document", 
    queries: List[str] = None,
    n_results: int = 3,
    semantic_weight: float = 0.7, 
    bm25_weight: float = 0.3
):
    """
    Retrieve similar chunks
    Args:
        collection_name: Name of the Chroma collection
        persist_directory: Directory to persist the vector database
        pdf_path: Path to the PDF file to process
        document_title: Title for the document
        queries: List of queries to test (optional)
        n_results: Number of results to return per query
        semantic_weight: Weight for semantic search
        bm25_weight: Weight for BM25 search
    """


    if queries is None:
        queries = [
            "What are the executive officers?",
            "financial performance revenue growth",
            "risk factors and challenges"
        ]
    
    # Step 2: Initialize retriever and query
    print("\nQUERYING DOCUMENTS")
    
    retriever = HybridRetriever(
        collection_name=collection_name,  
        persist_directory=persist_directory,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight
    )

    # Process each query
    for query_idx, query in enumerate(queries, 1):
        print(f"\n Query {query_idx}: {query}")
        
        try:
            results = retriever.retrieve(query, n_results=n_results)
            
            if not results:
                print(" No results found")
                continue
            
            for i, chunk in enumerate(results, 1):
                print(f"\n📋 Result {i}:")
                print(f"   Chunk ID: {chunk.chunk_id}")
                print(f"   Section: {chunk.metadata.get('section_title', 'Unknown')}")
                print(f"   Document: {chunk.metadata.get('document_title', 'Unknown')}")
                print(f"   Page(s): {chunk.metadata.get('page_numbers', 'Unknown')}")
                print(f"   Scores - Semantic: {chunk.semantic_score:.3f}, BM25: {chunk.bm25_score:.3f}, Hybrid: {chunk.hybrid_score:.3f}")
                print(f"   Content : {chunk.content}...")
                
        except Exception as e:
            print(f" Error processing query '{query}': {e}")
    
    # Print retriever stats
    print("\n📈 RETRIEVER STATISTICS")
    retriever_stats = retriever.get_retriever_stats()
    for key, value in retriever_stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    config = {
        "collection_name": "financial_docs",
        "persist_directory": "./chroma_db",
        "pdf_path": "C:/Users/tejup/Downloads/extraction_purpose2.pdf",
        "document_title": "Financial Report 2024",
        "queries": [
            "What are the executive officers?",
            "financial performance and revenue growth",
            "risk factors and challenges",
            "balance sheet assets and liabilities"
        ],
        "n_results": 3,
        "semantic_weight": 0.7,
        "bm25_weight": 0.3
    }
    
    try:
        embed(**config)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

