"""
Data models and types for the Dynamic RAG Pipeline
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import time

class ToolType(Enum):
    FINANCIAL_DATA = "financial_data"
    WEB_SEARCH = "web_search" 
    CALCULATOR = "calculator"

@dataclass
class QueryPreprocessingResult:
    """Result from query preprocessing agent"""
    is_valid_financial_query: bool
    normalized_query: str
    reasoning: str

@dataclass
class EvaluationResult:
    """Result from evaluator agent"""
    sufficient_context: bool
    missing_info: List[str]
    useful_info: str
    confidence_score: float
    evaluation_reasoning: str

@dataclass
class SubQuery:
    """Sub-query with tool and parameters"""
    subquery_id: str
    subquery: str
    tool: str
    depends_on: List[str]
    output_key: str 
    priority: str = "medium"
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class ToolResult:
    """Result from tool execution"""
    subquery_id: str
    tool_name: str
    result_data: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    @classmethod
    def success_result(cls, subquery_id: str, tool_name: str, data: Dict[str, Any], exec_time: float = 0.0):
        """Create a successful tool result"""
        return cls(
            subquery_id=subquery_id,
            tool_name=tool_name,
            result_data=data,
            success=True,
            execution_time=exec_time
        )
    
    @classmethod
    def failure_result(cls, subquery_id: str, tool_name: str, error: str, exec_time: float = 0.0):
        """Create a failed tool result"""
        return cls(
            subquery_id=subquery_id,
            tool_name=tool_name,
            result_data={},
            success=False,
            error_message=error,
            execution_time=exec_time
        )

@dataclass
class RetrievedChunk:
    """Retrieved chunk data structure"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: float
    bm25_score: float
    hybrid_score: float

@dataclass
class PipelineState:
    """State object passed between pipeline stages"""
    original_query: str
    normalized_query: str = ""
    is_valid_query: bool = True
    preprocessing_reasoning: str = ""
    
    # Retrieval results
    retrieved_chunks: List[RetrievedChunk] = None
    
    # Evaluation results
    sufficient_context: bool = False
    useful_info: str = ""
    missing_info: List[str] = None
    confidence_score: float = 0.0
    evaluation_reasoning: str = ""
    
    # Task planning results
    subqueries: List[SubQuery] = None
    
    # Tool execution results
    tool_results: List[ToolResult] = None
    
    # Final answer
    final_answer: str = ""
    
    def __post_init__(self):
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []
        if self.missing_info is None:
            self.missing_info = []
        if self.subqueries is None:
            self.subqueries = []
        if self.tool_results is None:
            self.tool_results = []

class RateLimitError(Exception):
    """Exception raised when API rate limit is hit"""
    pass

class PipelineError(Exception):
    """General pipeline error"""
    pass