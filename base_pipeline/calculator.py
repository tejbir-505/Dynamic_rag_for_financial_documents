import math 

eval_dict = {
    "abs": abs,
    "round": round,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e
}

class Calculator:
    def __init__(self):
        pass

    def calculate(self, expression: str) -> str:
        try:
            answer = eval(expression, {"__builtins__": {}}, eval_dict)
            return str(answer)
        except Exception as e:
            return f"error in calculation: {e}"
        
    def execute(self, expression):
        calc = Calculator()
        answer = calc.calculate(expression)
        return answer


### Yet to be coded ;)
# i am thinking of 


import json
import math
from typing import Dict, List, Any, Union
import logging

# Import your existing Calculator class
from your_calculator_module import Calculator  # Replace with actual import path

class CalculatorAgent:
    """
    Calculator Agent that processes subqueries requiring mathematical calculations
    and converts them into executable expressions using financial data.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the Calculator Agent
        
        Args:
            llm_client: Your LLM client (e.g., OpenAI, Anthropic, etc.)
        """
        self.calculator = Calculator()
        self.llm_client = llm_client
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load the calculator agent system prompt"""
        # You can load this from the artifact file or embed it directly
        return """
        # Calculator Agent System Prompt (embedded from artifact)
        # ... (include the full prompt from the artifact here)
        """
    
    def process_subquery(self, subquery: Dict, tool_results: List[Dict]) -> Dict:
        """
        Process a calculator subquery and return results
        
        Args:
            subquery: Dictionary containing subquery details
            tool_results: List of previous tool results this calculation depends on
            
        Returns:
            Dictionary with subquery_id, tool, output_key, and results
        """
        try:
            # Get dependent tool results
            dependent_data = self._extract_dependent_data(subquery, tool_results)
            
            # Generate calculation expressions using LLM
            calculation_plan = self._generate_calculation_plan(subquery, dependent_data)
            
            # Execute calculations
            results = self._execute_calculations(calculation_plan)
            
            # Format results
            formatted_results = self._format_results(results, calculation_plan)
            
            return {
                "subquery_id": subquery["subquery_id"],
                "tool": "calculator",
                "output_key": subquery["output_key"],
                "results": formatted_results
            }
            
        except Exception as e:
            logging.error(f"Error processing calculator subquery {subquery['subquery_id']}: {e}")
            return {
                "subquery_id": subquery["subquery_id"],
                "tool": "calculator",
                "output_key": subquery["output_key"],
                "results": {"error": str(e)}
            }
    
    def _extract_dependent_data(self, subquery: Dict, tool_results: List[Dict]) -> Dict:
        """Extract data from dependent tool results"""
        dependent_data = {}
        
        if "depends_on" in subquery:
            for dep_id in subquery["depends_on"]:
                for result in tool_results:
                    if result["subquery_id"] == dep_id:
                        dependent_data[dep_id] = result["results"]
                        break
        
        return dependent_data
    
    def _generate_calculation_plan(self, subquery: Dict, dependent_data: Dict) -> Dict:
        """
        Use LLM to generate calculation expressions based on subquery and data
        """
        # Prepare the prompt
        user_prompt = f"""
        Please analyze this calculation subquery and generate the mathematical expressions needed.

        **Subquery Details:**
        {json.dumps(subquery, indent=2)}

        **Available Data from Previous Tools:**
        {json.dumps(dependent_data, indent=2)}

        Generate the appropriate mathematical expressions to fulfill this calculation requirement.
        """
        
        if self.llm_client:
            # Use your LLM client here
            response = self.llm_client.generate(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                response_format="json"
            )
            return json.loads(response)
        else:
            # Fallback: Basic pattern matching for common financial calculations
            return self._fallback_calculation_plan(subquery, dependent_data)
    
    def _fallback_calculation_plan(self, subquery: Dict, dependent_data: Dict) -> Dict:
        """
        Fallback method when LLM is not available - handles common calculation patterns
        """
        calc_type = subquery.get("parameters", {}).get("calculation_type", "")
        expressions = []
        descriptions = []
        
        if calc_type == "percentage_change":
            # Handle percentage change calculations
            for dep_id, data in dependent_data.items():
                if isinstance(data, dict):
                    for ticker, values in data.items():
                        if isinstance(values, dict) and "current_price" in values and "fifty_day_average" in values:
                            expr = f"(({values['current_price']} - {values['fifty_day_average']}) / {values['fifty_day_average']}) * 100"
                            expressions.append(expr)
                            descriptions.append(f"{ticker} percentage change vs 50-day average")
        
        elif calc_type == "average":
            # Handle average calculations
            for dep_id, data in dependent_data.items():
                if isinstance(data, list):
                    values_str = " + ".join(map(str, data))
                    expr = f"({values_str}) / {len(data)}"
                    expressions.append(expr)
                    descriptions.append(f"Average of {len(data)} values")
        
        return {
            "expressions": expressions,
            "descriptions": descriptions,
            "context": f"Performing {calc_type} calculations"
        }
    
    def _execute_calculations(self, calculation_plan: Dict) -> List[Dict]:
        """Execute the mathematical expressions"""
        results = []
        
        expressions = calculation_plan.get("expressions", [])
        descriptions = calculation_plan.get("descriptions", [])
        
        for i, expression in enumerate(expressions):
            try:
                result = self.calculator.calculate(expression)
                results.append({
                    "expression": expression,
                    "result": result,
                    "description": descriptions[i] if i < len(descriptions) else f"Calculation {i+1}"
                })
            except Exception as e:
                results.append({
                    "expression": expression,
                    "result": None,
                    "error": str(e),
                    "description": descriptions[i] if i < len(descriptions) else f"Calculation {i+1}"
                })
        
        return results
    
    def _format_results(self, results: List[Dict], calculation_plan: Dict) -> Dict:
        """Format the calculation results for output"""
        return {
            "context": calculation_plan.get("context", "Mathematical calculations"),
            "calculations": results,
            "summary": self._generate_summary(results)
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate a summary of calculation results"""
        total_calculations = len(results)
        successful = sum(1 for r in results if r.get("result") is not None)
        failed = total_calculations - successful
        
        return {
            "total_calculations": total_calculations,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_calculations) * 100 if total_calculations > 0 else 0
        }

# Example usage and integration function
def process_calculator_subqueries(subqueries: List[Dict], tool_results: List[Dict], llm_client=None) -> List[Dict]:
    """
    Process multiple calculator subqueries
    
    Args:
        subqueries: List of subqueries that require calculator tool
        tool_results: List of all previous tool results
        llm_client: Your LLM client instance
        
    Returns:
        List of calculator results to be added to tool_results
    """
    calculator_agent = CalculatorAgent(llm_client)
    calculator_results = []
    
    for subquery in subqueries:
        if subquery.get("tool") == "calculator":
            result = calculator_agent.process_subquery(subquery, tool_results)
            calculator_results.append(result)
            # Add to tool_results so subsequent calculations can use this data
            tool_results.append(result)
    
    return calculator_results

# Example of how to integrate into your main pipeline
if __name__ == "__main__":
    # Example subquery
    example_subquery = {
        "subquery_id": "q4",
        "subquery": "Calculate percentage changes in bank stock prices around Fed decision",
        "tool": "calculator",
        "depends_on": ["q3"],
        "output_key": "bank_performance_analysis",
        "parameters": {
            "calculation_type": "percentage_change",
            "metrics": "price_change_analysis,volatility_comparison"
        }
    }
    
    # Example tool results from previous steps
    example_tool_results = [
        {
            "subquery_id": "q3",
            "tool": "financial_data",
            "output_key": "bank_stock_performance",
            "results": {
                "JPM": {"current_price": 150.25, "fifty_day_average": 145.80},
                "BAC": {"current_price": 35.60, "fifty_day_average": 34.20},
                "WFC": {"current_price": 42.15, "fifty_day_average": 41.00}
            }
        }
    ]
    
    # Process the calculator subquery
    calculator_agent = CalculatorAgent()
    result = calculator_agent.process_subquery(example_subquery, example_tool_results)
    
    print("Calculator Result:")
    print(json.dumps(result, indent=2))