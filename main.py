"""
Main entry point for the AI Agentic Research System.
"""

import logging
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
from config.settings import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

# Import the workflow
from config.workflow import create_workflow
from models.state import AgentState

class ResearchSystem:
    """
    Interface class for the AI Agentic Research System.
    Handles query processing.
    """

    def __init__(self):
        self.app = create_workflow()

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the agent system and return the results."""
        initial_state = AgentState(query=query)
        
        logger.info(f"Processing query: {query}")

        try:
            result = self.app.invoke(initial_state)
            
            response = {
                "query": query,
                "answer": getattr(result, "final_answer", "Unable to generate an answer."),
                "error": getattr(result, "error", None),
                "research_queries": [],
                "sources_count": 0,
            }

            if hasattr(result, "research_results") and result.research_results:
                response["research_queries"] = [item.get("query", "") for item in result.research_results]
                response["sources_count"] = sum(len(r.get("results", [])) for r in result.research_results)

            logger.info(f"Query processed successfully: {query[:50]}...")
            return response

        except Exception as e:
            logger.exception("Error processing query")
            return {
                "query": query,
                "answer": "An error occurred while processing your query.",
                "error": str(e),
                "research_queries": [],
                "sources_count": 0,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agentic Research System")
    parser.add_argument("--query", type=str, required=False, help="Query to process")
    args = parser.parse_args()

    # Instantiate the system
    system = ResearchSystem()

    if args.query:
        query = args.query
    else:
        query = "What are the latest advancements in quantum computing?"

    print(f"Processing query: {query}")
    
    result = system.process_query(query)
    
    print("\n--- Query Result ---")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    
    if result.get("error"):
        print(f"Error: {result['error']}")
    
    print("\n--- Research Statistics ---")
    print(f"- Search queries used: {result['research_queries']}")
    print(f"- Sources referenced: {result['sources_count']}")
