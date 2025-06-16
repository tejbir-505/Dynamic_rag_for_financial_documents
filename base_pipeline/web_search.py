import json
import time
from typing import Dict, List, Any
import logging
from duckduckgo_search import DDGS

class DuckDuckGoSearchTool:
    """
    DuckDuckGo web search tool using duckduckgo-search library
    Returns search results in standardized format for RAG pipeline
    """
    def __init__(self, max_results: int = 10, delay: float = 1.0):
        self.max_results = max_results
        self.delay = delay
        self.ddgs = DDGS()
        
    def search_web(self, query: str) -> Dict[str, Any]:
        try:
            results = list(self.ddgs.text(query, max_results=self.max_results))
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", ""),
                    "domain": self._extract_domain(result.get("href", ""))
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "search_type": "web"
            }
            
        except Exception as e:
            logging.error(f"Error in web search: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "search_type": "web",
                "error": str(e)
            }
    
    def search_news(self, query: str) -> Dict[str, Any]:
        try:
            results = list(self.ddgs.news(query, max_results=self.max_results))
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("url", ""),
                    "domain": self._extract_domain(result.get("url", "")),
                    "date": result.get("date", ""),
                    "source": result.get("source", "")
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "search_type": "news"
            }
            
        except Exception as e:
            logging.error(f"Error in news search: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "search_type": "news",
                "error": str(e)
            }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ""
    
    def process_subquery(self, subquery: Dict) -> Dict:
        """
        Process a web search subquery and return standardized results
        
        Args:
            subquery: Dictionary containing subquery details
            Format: {
                "subquery_id": "q1",
                "subquery": "Search for latest Federal Reserve interest rate decision",
                "tool": "web_search",
                "depends_on": [],
                "output_key": "fed_rate_decision",
                "parameters": {
                    "search_terms": "Federal Reserve interest rate decision December 2024"
                }
            }
            
        Returns:
            Dictionary in format: {
                "subquery_id": subquery["subquery_id"],
                "tool": "web_search",
                "output_key": subquery["output_key"],
                "results": formatted_results
            }
        """
        try:
            search_terms = subquery.get("parameters", {}).get("search_terms", "")
            if not search_terms:
                search_terms = subquery.get("subquery", "")
            
            # Determine search type
            search_type = "web"  
            if any(keyword in search_terms.lower() for keyword in ["news", "latest", "recent", "breaking"]):
                search_type = "news"
            
            if search_type == "news":
                search_results = self.search_news(search_terms)
            else:
                search_results = self.search_web(search_terms)
            
            formatted_results = self._format_for_rag(search_results, subquery)
            
            time.sleep(self.delay)
            
            return {
                "subquery_id": subquery["subquery_id"],
                "tool": "web_search",
                "output_key": subquery["output_key"],
                "results": formatted_results
            }
            
        except Exception as e:
            logging.error(f"Error processing web search subquery {subquery.get('subquery_id', 'unknown')}: {e}")
            return {
                "subquery_id": subquery["subquery_id"],
                "tool": "web_search",
                "output_key": subquery["output_key"],
                "results": {"error": str(e)}
            }
    
    def _format_for_rag(self, search_results: Dict, subquery: Dict) -> Dict:
        """
        Format search results for RAG pipeline 
        """
        results = search_results.get("results", [])
        
        summaries = []
        sources = []
        
        for i, result in enumerate(results):
            summaries.append({
                "title": result.get("title", ""),
                "content": result.get("snippet", ""),
                "url": result.get("url", ""),
                "domain": result.get("domain", ""),
                "relevance_score": self._calculate_relevance(result, subquery),
                "rank": i + 1
            })
            
            sources.append({
                "url": result.get("url", ""),
                "domain": result.get("domain", ""),
                "title": result.get("title", ""),
                "date": result.get("date", ""),
                "source": result.get("source", "")
            })
        
        return {
            "search_query": search_results.get("query", ""),
            "total_results": search_results.get("total_results", 0),
            "search_type": search_results.get("search_type", "web"),
            "summaries": summaries,
            "sources": sources,
            "search_metadata": {
                "timestamp": time.time(),
                "subquery_purpose": subquery.get("subquery", ""),
                "search_successful": len(results) > 0,
                "has_error": "error" in search_results
            }
        }
    
    def _calculate_relevance(self, result: Dict, subquery: Dict) -> float:
        """
        Simple relevance scoring based on keyword matching
        """
        try:
            query_words = subquery.get("subquery", "").lower().split()
            result_text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
            
            matches = sum(1 for word in query_words if word in result_text and len(word) > 3)
            return round(matches / len(query_words), 2) if query_words else 0.0
            
        except:
            return 0.5  

def process_web_search_subqueries(subqueries: List[Dict]) -> List[Dict]:
    """
    Process multiple web search subqueries
    
    Args:
        subqueries: List of subqueries that require web_search tool
        
    Returns:
        List of web search results to be added to tool_results
    """
    search_tool = DuckDuckGoSearchTool()
    web_search_results = []
    
    for subquery in subqueries:
        if subquery.get("tool") == "web_search":
            result = search_tool.process_subquery(subquery)
            web_search_results.append(result)
    
    return web_search_results




# if __name__ == "__main__":
#     example_subqueries = [
#         {
#             "subquery_id": "q1",
#             "subquery": "Search for latest Federal Reserve interest rate decision and details",
#             "tool": "web_search",
#             "depends_on": [],
#             "output_key": "fed_rate_decision",
#             "parameters": {
#                 "search_terms": "Federal Reserve interest rate decision December 2024 FOMC meeting"
#             }
#         },
#         {
#             "subquery_id": "q2",
#             "subquery": "Search for banking sector reaction and market analysis to Fed decision",
#             "tool": "web_search",
#             "depends_on": [],
#             "output_key": "banking_sector_reaction",
#             "parameters": {
#                 "search_terms": "banking stocks reaction Fed rate decision December 2024 market impact"
#             }
#         }
#     ]

#     search_tool = DuckDuckGoSearchTool()
    
#     for subquery in example_subqueries:
#         result = search_tool.process_subquery(subquery)
#         print(f"\nWeb Search Result for {subquery['subquery_id']}:")
#         print(json.dumps(result, indent=2))