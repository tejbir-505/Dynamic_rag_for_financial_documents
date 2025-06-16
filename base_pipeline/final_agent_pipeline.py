"""
Main Dynamic RAG Pipeline using LangGraph
"""
import logging
import os
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent_data import PipelineState, RetrievedChunk, RateLimitError, PipelineError
from agents_initialisation import (
    QueryPreprocessingAgent, EvaluatorAgent, 
    TaskPlannerAgent, FinalAnswerGeneratorAgent
)
from tool_executor import ToolExecutor

try:
    from retriever import HybridRetriever
except ImportError:
    logging.warning("Retriever module not found. Please ensure retriever.py is available.")
    HybridRetriever = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicRAGPipeline:
    """Main pipeline orchestrating all agents and tools"""
    
    def __init__(self, collection_name: str, persist_directory: str, 
                 semantic_weight: float = 0.7, bm25_weight: float = 0.3, 
                 n_results: int = 3):        
        # Store retrieval parameters
        self.n_results = n_results
        
        # Initialize agents
        try:
            self.query_preprocessor = QueryPreprocessingAgent()
            self.evaluator = EvaluatorAgent()
            self.task_planner = TaskPlannerAgent()
            self.answer_generator = FinalAnswerGeneratorAgent()
            self.tool_executor = ToolExecutor()
            logger.info("Successfully initialized all agents")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise PipelineError(f"Agent initialization failed: {e}")
        
        # Initialize retriever
        if HybridRetriever is None:
            logger.error("HybridRetriever not available")
            raise PipelineError("Retriever module not found")
            
        try:
            self.retriever = HybridRetriever(
                collection_name=collection_name,
                persist_directory=persist_directory,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight
            )
            logger.info("Successfully initialized retriever")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise PipelineError(f"Retriever initialization failed: {e}")
        
        # Build the graph
        self.graph = self._build_graph()
        logger.info("Pipeline initialized successfully")
        
    def _build_graph(self) -> StateGraph:
        """Building the langgraph workflow"""
        
        workflow = StateGraph(PipelineState)
        
        workflow.add_node("preprocess_query", self._preprocess_query)
        workflow.add_node("retrieve_chunks", self._retrieve_chunks)
        workflow.add_node("evaluate_context", self._evaluate_context)
        workflow.add_node("plan_tasks", self._plan_tasks)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("generate_answer", self._generate_answer)
        
        workflow.set_entry_point("preprocess_query")
        
        # Conditional routing after preprocessing
        workflow.add_conditional_edges(
            "preprocess_query",
            self._should_continue_after_preprocessing,
            {
                "continue": "retrieve_chunks",
                "end": END
            }
        )
        
        workflow.add_edge("retrieve_chunks", "evaluate_context")
        
        # Conditional routing after evaluation
        workflow.add_conditional_edges(
            "evaluate_context",
            self._should_use_tools,
            {
                "sufficient": "generate_answer",
                "insufficient": "plan_tasks"
            }
        )
        
        workflow.add_edge("plan_tasks", "execute_tools")
        workflow.add_edge("execute_tools", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _preprocess_query(self, state: PipelineState) -> Dict[str, Any]:
        """Node: Preprocess and validate the user query"""
        logger.info("Starting query preprocessing...")
        
        try:
            result = self.query_preprocessor.process(state.original_query)
            
            # state.normalized_query = result.normalized_query
            # state.is_valid_query = result.is_valid_financial_query
            # state.preprocessing_reasoning = result.reasoning
            
            # logger.info(f"Query preprocessing completed. Valid: {state.is_valid_query}")
            
            # print(f"normalized_query {state.normalized_query}")
            # print(f"is_valid_query {state.is_valid_query}")
            # print(f"preprocessing_reasoning {state.preprocessing_reasoning}")
            return {
                result
            }
            
        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            state.is_valid_query = False
            state.preprocessing_reasoning = f"Preprocessing error: {str(e)}"
            
            return {
                "normalized_query": state.original_query,
                "is_valid_query": False,
                "preprocessing_reasoning": state.preprocessing_reasoning
            }
    
    def _retrieve_chunks(self, state: PipelineState) -> Dict[str, Any]:
        """Node: Retrieve relevant chunks"""
        logger.info("Starting chunk retrieval...")
        
        try:
            query_to_use = state.normalized_query if state.normalized_query else state.original_query
            
            results = self.retriever.retrieve(query_to_use, n_results=self.n_results)
            
            # Convert results to our data structure if needed
            retrieved_chunks = []
            for chunk in results:
                if hasattr(chunk, 'chunk_id'):
                    # Already in correct format
                    retrieved_chunks.append(chunk)
                else:
                    # Convert if needed
                    retrieved_chunk = RetrievedChunk(
                        chunk_id=getattr(chunk, 'chunk_id', f"chunk_{len(retrieved_chunks)}"),
                        content=getattr(chunk, 'content', str(chunk)),
                        metadata=getattr(chunk, 'metadata', {}),
                        semantic_score=getattr(chunk, 'semantic_score', 0.0),
                        bm25_score=getattr(chunk, 'bm25_score', 0.0),
                        hybrid_score=getattr(chunk, 'hybrid_score', 0.0)
                    )
                    retrieved_chunks.append(retrieved_chunk)
            
            state.retrieved_chunks = retrieved_chunks
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            
            return {
                "retrieved_chunks": state.retrieved_chunks
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")

            print(f" RETRIEVER FAILED: {str(e)}")
            print(" Pipeline ended due to retrieval failure.")
            
            # Set empty chunks and mark as failed
            state.retrieved_chunks = []
            state.final_answer = f"I apologize, but I cannot process your query due to a retrieval system failure: {str(e)}"
            
            return {
                "retrieved_chunks": [],
                "final_answer": state.final_answer
            }
    
    def _evaluate_context(self, state: PipelineState) -> Dict[str, Any]:
        """Node: Evaluate if retrieved context is sufficient"""
        logger.info("Starting context evaluation")
        
        try:
            # If no chunks retrieved due to failure, mark as insufficient
            if not state.retrieved_chunks:
                state.sufficient_context = False
                state.useful_info = ""
                state.missing_info = ["Unable to retrieve any relevant information"]
                state.confidence_score = 0.0
                state.evaluation_reasoning = "No chunks retrieved"
                
                logger.warning("No chunks available for evaluation")
                
                return {
                    "sufficient_context": False,
                    "useful_info": "",
                    "missing_info": state.missing_info,
                    "confidence_score": 0.0,
                    "evaluation_reasoning": state.evaluation_reasoning
                }
            
            result = self.evaluator.evaluate(
                state.original_query,
                state.normalized_query,
                state.retrieved_chunks
            )
            
            # Update state
            state.sufficient_context = result.sufficient_context
            state.useful_info = result.useful_info
            state.missing_info = result.missing_info
            state.confidence_score = result.confidence_score
            state.evaluation_reasoning = result.evaluation_reasoning
            
            logger.info(f"Context evaluation completed. Sufficient: {state.sufficient_context}")
            
            return {
                "sufficient_context": state.sufficient_context,
                "useful_info": state.useful_info,
                "missing_info": state.missing_info,
                "confidence_score": state.confidence_score,
                "evaluation_reasoning": state.evaluation_reasoning
            }
            
        except Exception as e:
            logger.error(f"Context evaluation failed: {e}")
            # Default to insufficient context if evaluation fails
            state.sufficient_context = False
            state.useful_info = ""
            state.missing_info = [f"Evaluation error: {str(e)}"]
            state.confidence_score = 0.0
            state.evaluation_reasoning = f"Evaluation failed: {str(e)}"
            
            return {
                "sufficient_context": False,
                "useful_info": "",
                "missing_info": state.missing_info,
                "confidence_score": 0.0,
                "evaluation_reasoning": state.evaluation_reasoning
            }
    
    def _plan_tasks(self, state: PipelineState) -> Dict[str, Any]:
        """Node: Plan tasks and create sub-queries"""
        logger.info("Starting task planning...")
        
        try:
            subqueries = self.task_planner.plan(
                state.original_query,
                state.normalized_query,
                state.useful_info,
                state.missing_info
            )
            
            state.subqueries = subqueries
            
            logger.info(f"Task planning completed. Generated {len(subqueries)} sub-queries")
            
            return {
                "subqueries": state.subqueries
            }
            
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            state.subqueries = []
            
            return {
                "subqueries": []
            }
    
    def _execute_tools(self, state: PipelineState) -> Dict[str, Any]:
        """Node: Execute tools in parallel"""
        logger.info("Starting tool execution...")
        
        try:
            if not state.subqueries:
                logger.warning("No sub-queries to execute")
                state.tool_results = []
                
                return {
                    "tool_results": []
                }
            
            # Execute sub-queries using tool executor
            tool_results = self.tool_executor.execute_subqueries(state.subqueries)
            
            state.tool_results = tool_results
            
            # Log execution summary
            successful_tools = len([r for r in tool_results if r.success])
            failed_tools = len(tool_results) - successful_tools
            
            logger.info(f"Tool execution completed. Success: {successful_tools}, Failed: {failed_tools}")
            
            return {
                "tool_results": state.tool_results
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            state.tool_results = []
            
            return {
                "tool_results": []
            }
    
    def _generate_answer(self, state: PipelineState) -> Dict[str, Any]:
        """Node: Generate final answer"""
        logger.info("Starting final answer generation...")
        
        try:
            # If retrieval failed and we already have an error message, use it
            if state.final_answer and "retrieval system failure" in state.final_answer:
                return {
                    "final_answer": state.final_answer
                }
            
            # Ensure we have tool_results list (empty if not set)
            if state.tool_results is None:
                state.tool_results = []
            
            final_answer = self.answer_generator.generate_answer(
                state.original_query,
                state.normalized_query,
                state.retrieved_chunks,
                state.useful_info,
                state.confidence_score,
                state.tool_results
            )
            
            state.final_answer = final_answer
            
            logger.info("Final answer generation completed")
            
            return {
                "final_answer": state.final_answer
            }
            
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            state.final_answer = f"I apologize, but I encountered an error while generating the final answer: {str(e)}"
            
            return {
                "final_answer": state.final_answer
            }
    
    def _should_continue_after_preprocessing(self, state: PipelineState) -> Literal["continue", "end"]:
        """Conditional edge: Check if query is valid"""
        if not state.is_valid_query:
            print(f"❌ INVALID QUERY: {state.preprocessing_reasoning}")
            print("❌ Pipeline ended due to invalid query.")
            print(f"normalized_query {state.normalized_query}")
            print(f"is_valid_query {state.is_valid_query}")
            print(f"preprocessing_reasoning {state.preprocessing_reasoning}")
            return "end"
        return "continue"
    
    def _should_use_tools(self, state: PipelineState) -> Literal["sufficient", "insufficient"]:
        """Conditional edge: Check if context is sufficient"""
        if state.sufficient_context:
            logger.info("Context is sufficient, proceeding to answer generation")
            return "sufficient"
        else:
            logger.info("Context is insufficient, proceeding to task planning")
            return "insufficient"
    
    def run(self, query: str, config: Dict[str, Any] = None) -> str:
        """Run the complete pipeline"""
        logger.info(f"Starting pipeline for query: {query[:100]}...")
        
        # Create initial state
        initial_state = PipelineState(original_query=query)
        
        # Configure run
        if config is None:
            config = {"thread_id": "default"}
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state, config=config)
            
            # Return the final answer
            return final_state.get("final_answer", "No answer generated")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return f"Pipeline execution failed: {str(e)}"
    
    def run_with_state(self, query: str, config: Dict[str, Any] = None) -> PipelineState:
        """Run pipeline and return complete state"""
        logger.info(f"Starting pipeline with state return for query: {query[:100]}...")
        
        # Create initial state
        initial_state = PipelineState(original_query=query)
        
        # Configure run
        if config is None:
            config = {"thread_id": "default"}
        
        try:
            # Run the graph
            final_state_dict = self.graph.invoke(initial_state, config=config)
            
            # Convert back to PipelineState object
            final_state = PipelineState(original_query=query)
            
            # Update with all returned values
            for key, value in final_state_dict.items():
                if hasattr(final_state, key):
                    setattr(final_state, key, value)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            final_state = PipelineState(original_query=query)
            final_state.final_answer = f"Pipeline execution failed: {str(e)}"
            return final_state




# if __name__ == "__main__":
#     pipeline = DynamicRAGPipeline(
#         collection_name="financial_docs",
#         persist_directory="./chroma_db",
#         semantic_weight=0.7,
#         bm25_weight=0.3,
#         n_results=3
#     )

#     test_query = "How's Tesla doing lately?"
    
#     print("="*80)
#     print("🚀 STARTING DYNAMIC RAG PIPELINE")
#     print("="*80)
    
#     answer = pipeline.run(test_query)
    
#     print("\n" + "="*80)
#     print("✅ PIPELINE COMPLETED")
#     print("="*80)
#     print(f"Query: {test_query}")
#     print(f"Answer: {answer}")
    
#     # For debugging, you can also get the full state
#     print("\n" + "="*80)
#     print("🔍 FULL PIPELINE STATE")
#     print("="*80)
    
#     full_state = pipeline.run_with_state(test_query)
#     print(f"Valid Query: {full_state.is_valid_query}")
#     print(f"Normalized Query: {full_state.normalized_query}")
#     print(f"Retrieved Chunks: {len(full_state.retrieved_chunks)}")
#     print(f"Sufficient Context: {full_state.sufficient_context}")
#     print(f"Confidence Score: {full_state.confidence_score}")
#     print(f"Sub-queries Generated: {len(full_state.subqueries)}")
#     print(f"Tool Results: {len(full_state.tool_results)}")


pipeline2 = DynamicRAGPipeline(
    collection_name="financial_docs",
    persist_directory="./chroma_db",
    semantic_weight=0.7,
    bm25_weight=0.3,
    n_results=3
)
query = "Tell me about Meta's earnings"
initial_state = PipelineState(original_query=query)
answer = pipeline2._preprocess_query(initial_state)
print(answer)
