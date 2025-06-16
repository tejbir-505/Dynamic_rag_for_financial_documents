"""
Tool executor for parallel execution of sub-queries
"""

import asyncio
import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent_data import SubQuery, ToolResult

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Executes tools in parallel with dependency management"""
    
    def __init__(self):
        self.max_workers = 5  # Maximum parallel tool executions
        
    def execute_subqueries(self, subqueries: List[SubQuery]) -> List[ToolResult]:
        """Execute sub-queries with dependency management"""
        if not subqueries:
            return []
            
        results = {}
        executed_queries = set()
        
        # Sort by dependencies and priority
        sorted_queries = self._sort_by_dependencies(subqueries)
        
        # Execute in batches based on dependencies
        while sorted_queries:
            # Find queries that can be executed (no unmet dependencies)
            ready_queries = []
            remaining_queries = []
            
            for query in sorted_queries:
                if all(dep in executed_queries for dep in query.depends_on):
                    ready_queries.append(query)
                else:
                    remaining_queries.append(query)
            
            if not ready_queries:
                logger.error("Circular dependency detected in sub-queries")
                break
                
            # Execute ready queries in parallel
            batch_results = self._execute_batch(ready_queries, results)
            
            # Update results and executed set
            for result in batch_results:
                results[result.subquery_id] = result
                if result.success:
                    executed_queries.add(result.subquery_id)
                else:
                    logger.warning(f"Tool execution failed for {result.subquery_id}: {result.error_message}")
            
            sorted_queries = remaining_queries
            
        return list(results.values())
    
    def _sort_by_dependencies(self, subqueries: List[SubQuery]) -> List[SubQuery]:
        """Sort sub-queries by dependency order"""
        # Simple topological sort
        sorted_queries = []
        remaining = subqueries.copy()
        
        while remaining:
            # Find queries with no dependencies or all dependencies already sorted
            ready = []
            not_ready = []
            
            sorted_ids = {q.subquery_id for q in sorted_queries}
            
            for query in remaining:
                if all(dep in sorted_ids for dep in query.depends_on):
                    ready.append(query)
                else:
                    not_ready.append(query)
            
            if not ready:
                # Add remaining queries even if dependencies aren't met
                ready = not_ready
                not_ready = []
                
            # Sort by priority within ready queries
            priority_order = {"high": 0, "medium": 1, "low": 2}
            ready.sort(key=lambda x: priority_order.get(x.priority, 1))
            
            sorted_queries.extend(ready)
            remaining = not_ready
            
        return sorted_queries
    
    def _execute_batch(self, queries: List[SubQuery], previous_results: Dict[str, ToolResult]) -> List[ToolResult]:
        """Execute a batch of queries in parallel"""
        if not queries:
            return []
            
        results = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=min(len(queries), self.max_workers)) as executor:
            # Submit all queries
            future_to_query = {}
            for query in queries:
                future = executor.submit(self._execute_single_query, query, previous_results)
                future_to_query[future] = query
            
            # Collect results as they complete
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per query
                    results.append(result)
                except Exception as e:
                    logger.error(f"Tool execution failed for {query.subquery_id}: {e}")
                    results.append(ToolResult.failure_result(
                        query.subquery_id, 
                        query.tool,
                        f"Execution error: {str(e)}"
                    ))
        
        return results
    
    def _execute_single_query(self, query: SubQuery, previous_results: Dict[str, ToolResult]) -> ToolResult:
        """Execute a single sub-query"""
        start_time = time.time()
        
        try:
            if query.tool == "financial_data":
                result_data = self._execute_financial_data(query)
            elif query.tool == "web_search":
                result_data = self._execute_web_search(query)
            elif query.tool == "calculator":
                result_data = self._execute_calculator(query, previous_results)
            else:
                raise ValueError(f"Unknown tool: {query.tool}")
            
            execution_time = time.time() - start_time
            
            return ToolResult.success_result(
                query.subquery_id,
                query.tool,
                result_data,
                execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool {query.tool} failed for query {query.subquery_id}: {e}")
            
            return ToolResult.failure_result(
                query.subquery_id,
                query.tool,
                str(e),
                execution_time
            )
    
    def _execute_financial_data(self, query: SubQuery) -> Dict[str, Any]:
        """Execute financial data tool"""
        try:
            # Import your financial data module
            from financial_data_parser import process_task_planner_subquery
            
            result = process_task_planner_subquery(query.subquery)
            
            if isinstance(result, dict):
                return result
            else:
                return {"data": result}
                
        except ImportError:
            logger.error("Financial data module not found")
            return {"error": "Financial data module not available"}
        except Exception as e:
            raise Exception(f"Financial data tool error: {e}")
    
    def _execute_web_search(self, query: SubQuery) -> Dict[str, Any]:
        """Execute web search tool"""
        try:
            # Import your web search module
            from web_search import DuckDuckGoSearchTool
            
            search_tool = DuckDuckGoSearchTool()
            result = search_tool.process_subquery(query.subquery)
            
            if isinstance(result, dict):
                return result
            else:
                return {"search_results": result}
                
        except ImportError:
            logger.error("Web search module not found")
            return {"error": "Web search module not available"}
        except Exception as e:
            # For web search, continue with partial data
            logger.warning(f"Web search failed, continuing with partial data: {e}")
            return {"error": str(e), "partial_results": []}
    
    def _execute_calculator(self, query: SubQuery, previous_results: Dict[str, ToolResult]) -> Dict[str, Any]:
        """Execute calculator tool"""
        try:
            # Import your calculator module
            from calc_agent import CalculatorAgent # Got lazy with calculator agent 🦥
            
            # Prepare tool results for calculator
            tool_results_data = {}
            for result in previous_results.values():
                if result.success:
                    tool_results_data[result.subquery_id] = result.result_data
            
            calculator_agent = CalculatorAgent()
            result = calculator_agent.process_subquery(query.subquery, tool_results_data)
            
            if isinstance(result, dict):
                return result
            else:
                return {"calculation_result": result}
                
        except ImportError:
            logger.error("Calculator module not found")
            return {"error": "Calculator module not available"}
        except Exception as e:
            raise Exception(f"Calculator tool error: {e}")

# Async version for future use
class AsyncToolExecutor:
    """Async version of tool executor"""
    
    async def execute_subqueries_async(self, subqueries: List[SubQuery]) -> List[ToolResult]:
        """Execute sub-queries asynchronously"""
        if not subqueries:
            return []
            
        # Group queries by dependency levels
        dependency_levels = self._group_by_dependency_level(subqueries)
        
        all_results = {}
        
        # Execute each level in sequence, but queries within level in parallel
        for level_queries in dependency_levels:
            level_results = await self._execute_level_async(level_queries, all_results)
            all_results.update({r.subquery_id: r for r in level_results})
        
        return list(all_results.values())
    
    def _group_by_dependency_level(self, subqueries: List[SubQuery]) -> List[List[SubQuery]]:
        """Group queries by dependency level"""
        levels = []
        remaining = subqueries.copy()
        processed_ids = set()
        
        while remaining:
            current_level = []
            next_remaining = []
            
            for query in remaining:
                if all(dep in processed_ids for dep in query.depends_on):
                    current_level.append(query)
                    processed_ids.add(query.subquery_id)
                else:
                    next_remaining.append(query)
            
            if not current_level:
                # Handle circular dependencies by adding all remaining
                current_level = next_remaining
                next_remaining = []
                for query in current_level:
                    processed_ids.add(query.subquery_id)
            
            if current_level:
                levels.append(current_level)
            remaining = next_remaining
        
        return levels
    
    async def _execute_level_async(self, queries: List[SubQuery], previous_results: Dict[str, ToolResult]) -> List[ToolResult]:
        """Execute queries in a single dependency level"""
        tasks = []
        for query in queries:
            task = asyncio.create_task(self._execute_single_query_async(query, previous_results))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolResult.failure_result(
                    queries[i].subquery_id,
                    queries[i].tool,
                    str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_query_async(self, query: SubQuery, previous_results: Dict[str, ToolResult]) -> ToolResult:
        """Execute single query asynchronously"""
        # Convert sync execution to async using run_in_executor
        loop = asyncio.get_event_loop()
        executor = ToolExecutor()
        
        return await loop.run_in_executor(
            None, 
            executor._execute_single_query, 
            query, 
            previous_results
        )