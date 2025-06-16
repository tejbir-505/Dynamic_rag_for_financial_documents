"""
Agent implementations for the Dynamic RAG Pipeline
"""

import json
import os
import logging
from typing import List, Dict, Any
from groq import Groq
import google.generativeai as genai
from agent_data import (
    QueryPreprocessingResult, EvaluationResult, SubQuery, 
    RetrievedChunk, ToolResult, RateLimitError
)
from dotenv import load_dotenv
load_dotenv()

# Load prompts
try:
    from prompts import (
        QUERY_PREPROCESSING_AGENT_PROMPT,
        EVALUATOR_AGENT_PROMPT, 
        TASK_PLANNER_AGENT_PROMPT,
        FINAL_ANSWER_GENERATOR_PROMPT
    )
except ImportError:
    # Fallback if prompts module not available
    QUERY_PREPROCESSING_AGENT_PROMPT = "Query preprocessing prompt placeholder"
    EVALUATOR_AGENT_PROMPT = "Evaluator prompt placeholder"
    TASK_PLANNER_AGENT_PROMPT = "Task planner prompt placeholder"
    FINAL_ANSWER_GENERATOR_PROMPT = "Final answer prompt placeholder"

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        self.groq_client = Groq(api_key=self.groq_api_key)
        genai.configure(api_key=self.gemini_api_key)
        
    def _call_groq(self, prompt: str, model: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    if attempt == max_retries - 1:
                        raise RateLimitError(f"Groq API rate limit exceeded: {e}")
                    continue
                else:
                    logger.error(f"Groq API error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise e
        
    def _call_gemini(self, prompt: str, model_name: str = "gemini-1.5-flash", max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4000
                    )
                )
                return response.text
            except Exception as e:
                if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                    if attempt == max_retries - 1:
                        raise RateLimitError(f"Gemini API rate limit exceeded: {e}")
                    continue
                else:
                    logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise e

class QueryPreprocessingAgent(BaseAgent):
    """Query preprocessing and validation agent"""

    def __init__(self):
        super().__init__()
        self.model = "llama-3.3-70b-versatile"
        
    def process(self, user_query: str) -> QueryPreprocessingResult:
        try:
            prompt = QUERY_PREPROCESSING_AGENT_PROMPT.format(user_query=user_query)
            response = self._call_groq(prompt, self.model)
            
            # Parse JSON response
            # result_dict = json.loads(response.strip())
            result_dict = self._extract_json_from_response(response.strip())

            return QueryPreprocessingResult(
                is_valid_financial_query=result_dict["is_valid_financial_query"],
                normalized_query=result_dict["normalized_query"],
                reasoning=result_dict["reasoning"]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse query preprocessing response: {e}")
            return QueryPreprocessingResult(
                is_valid_financial_query=True,
                normalized_query=user_query,
                reasoning="Failed to parse agent response, using original query"
            )
        except Exception as e:
            logger.error(f"Query preprocessing error: {e}")
            raise

class EvaluatorAgent(BaseAgent):    
    def __init__(self):
        super().__init__()
        self.model = "llama-3.3-70b-versatile"
        
    def evaluate(self, original_query: str, normalized_query: str, 
                retrieved_chunks: List[RetrievedChunk]) -> EvaluationResult:
        try:
            # Format retrieved chunks for prompt
            chunks_text = ""

            ### I might wanna change the chunk text format later if sectiona dn relevance score isn't relevant
            for i, chunk in enumerate(retrieved_chunks, 1):
                chunks_text += f"\n--- Chunk {i} ---\n"
                chunks_text += f"Source: {chunk.metadata.get('document_title', 'Unknown')} "
                chunks_text += f"(Page {chunk.metadata.get('page_numbers', 'Unknown')})\n"
                chunks_text += f"Section: {chunk.metadata.get('section_title', 'Unknown')}\n"
                chunks_text += f"Content: {chunk.content}\n"
                chunks_text += f"Relevance Score: {chunk.hybrid_score:.3f}\n"
            
            prompt = EVALUATOR_AGENT_PROMPT.format(
                original_query=original_query,
                normalized_query=normalized_query,
                retrieved_chunks=chunks_text
            )
            
            response = self._call_groq(prompt, self.model)
            result_dict = json.loads(response.strip())
            
            return EvaluationResult(
                sufficient_context=result_dict["sufficient_context"],
                missing_info=result_dict["missing_info"],
                useful_info=result_dict["useful_info"],
                confidence_score=result_dict["confidence_score"],
                evaluation_reasoning=result_dict["evaluation_reasoning"]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluator response: {e}")
            # Conservative fallback - assume insufficient context
            return EvaluationResult(
                sufficient_context=False,
                missing_info=["Unable to evaluate context sufficiency"],
                useful_info="",
                confidence_score = 0.5,
                evaluation_reasoning="Failed to evaluate agent response"
            )
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise

class TaskPlannerAgent(BaseAgent):
    """Task planning agent"""
    
    def __init__(self):
        super().__init__()
        self.model = "gemini-1.5-flash"
        
    def plan(self, original_query: str, normalized_query: str, 
             useful_info: str, missing_info: List[str]) -> List[SubQuery]:
        try:
            prompt = TASK_PLANNER_AGENT_PROMPT.format(
                original_query=original_query,
                normalized_query=normalized_query,
                useful_info=useful_info,
                missing_info=", ".join(missing_info)
            )
            
            response = self._call_gemini(prompt, self.model)
            
            # Parse JSON response
            subqueries_list = json.loads(response.strip())
            
            subqueries = []
            for sq_dict in subqueries_list:
                subquery = SubQuery(
                    subquery_id=sq_dict["subquery_id"],
                    subquery=sq_dict["subquery"],
                    tool=sq_dict["tool"],
                    depends_on=sq_dict.get("depends_on", []),
                    output_key=sq_dict["output_key"],
                    priority=sq_dict.get("priority", "medium"),
                    parameters=sq_dict.get("parameters", {})
                )
                subqueries.append(subquery)
                
            return subqueries
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse task planner response: {e}")
            return []
        except Exception as e:
            logger.error(f"Task planning error: {e}")
            raise

class FinalAnswerGeneratorAgent(BaseAgent):
    """Final answer generation agent"""
    
    def __init__(self):
        super().__init__()
        self.model = "gemini-1.5-flash"
        
    def generate_answer(self, original_query: str, normalized_query: str,
                       retrieved_chunks: List[RetrievedChunk], useful_info: str,
                       confidence_score: float, tool_results: List[ToolResult]) -> str:
        """Generate final comprehensive answer"""
        try:
            # Format retrieved chunks
            chunks_text = ""
            for i, chunk in enumerate(retrieved_chunks, 1):
                chunks_text += f"\n--- Chunk {i} ---\n"
                chunks_text += f"Source: {chunk.metadata.get('document_title', 'Unknown')}\n"
                chunks_text += f"Content: {chunk.content[:500]}...\n"
            
            # Format tool results
            tool_results_text = ""
            for result in tool_results:
                if result.success:
                    tool_results_text += f"\n--- {result.tool_name.upper()} Results ---\n"
                    tool_results_text += f"Query: {result.subquery_id}\n"
                    tool_results_text += f"Data: {json.dumps(result.result_data, indent=2)}\n"
                else:
                    tool_results_text += f"\n--- {result.tool_name.upper()} (FAILED) ---\n"
                    tool_results_text += f"Error: {result.error_message}\n"
            
            prompt = FINAL_ANSWER_GENERATOR_PROMPT.format(
                original_query=original_query,
                normalized_query=normalized_query,
                retrieved_chunks=chunks_text,
                useful_info=useful_info,
                confidence_score=confidence_score,
                tool_results=tool_results_text
            )
            
            response = self._call_gemini(prompt, self.model)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Final answer generation error: {e}")
            return f"Encountered an error while generating the final answer: {str(e)}"