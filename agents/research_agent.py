"""
Research Agent implementation for the AI Agentic Research System.
Responsible for gathering information from the web using Tavily.
"""
from typing import List, Dict, Any, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain.agents.tool_executor import ToolExecutor
from agents.utils import SimpleToolExecutor as ToolExecutor
from models.state import AgentState
from config.settings import get_settings
from agents.utils import format_error

# Get logger
logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    Agent responsible for gathering information from the web using Tavily.
    """
    def __init__(self, llm=None):
        settings = get_settings()
        
        # Initialize LLM
        self.llm = llm or ChatOpenAI(
            model=settings.default_model,
            temperature=settings.research_agent_temperature
        )
        
        # Initialize tools
        self.search_tool = TavilySearchResults(
            k=settings.max_search_results_per_query
        )
        self.tools = [self.search_tool]
        self.tool_executor = ToolExecutor(self.tools)
        
        # Setup the query generation chain
        self.query_generation_prompt = ChatPromptTemplate.from_template(
            """
            You are a research agent. Your goal is to gather comprehensive information about the following query:
            
            QUERY: {query}
            
            Based on this query, what are the top {num_search_queries} specific search queries you should make to gather 
            the most relevant and complete information? Be specific and targeted in your search queries.
            
            Output should be in the following JSON format:
            {{
                "search_queries": [
                    "specific search query 1",
                    "specific search query 2",
                    ...
                ],
                "reasoning": "Your reasoning for these search queries"
            }}
            """
        )
        self.query_generation_chain = self.query_generation_prompt | self.llm | JsonOutputParser()
    
    def process(self, state: AgentState) -> AgentState:
        """
        Process the state and gather research information.
        
        Args:
            state: Current state of the agent system
            
        Returns:
            Updated state with research results
        """
        logger.info(f"Research agent processing query: {state.query}")
        settings = get_settings()
        
        try:
            # Generate search queries
            search_queries_result = self.query_generation_chain.invoke({
                "query": state.query,
                "num_search_queries": settings.num_search_queries
            })
            
            logger.info(f"Generated {len(search_queries_result['search_queries'])} search queries")
            
            search_results = []
            
            # Execute the search for each query
            for query in search_queries_result["search_queries"]:
                logger.info(f"Executing search for: {query}")
                tool_input = {
                    "query": query, 
                    "max_results": settings.max_search_results_per_query
                }
                
                result = self.tool_executor.invoke({
                    "tool_name": "tavily_search_results_json", 
                    "tool_input": tool_input
                })
                
                search_results.append({
                    "query": query, 
                    "results": result
                })
                
                logger.info(f"Search completed for '{query}', found {len(result)} results")
            
            # Update the state with research results
            state.research_results = search_results
            
            # Add intermediate step
            state.add_intermediate_step(
                agent_name="research_agent",
                action="search",
                details={
                    "search_queries": search_queries_result,
                    "results_summary": f"Found {sum(len(r['results']) for r in search_results)} results from {len(search_results)} queries"
                }
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Error in research agent: {str(e)}")
            error_details = format_error(e)
            
            # Update state with error
            state.error = f"Research agent error: {error_details}"
            
            # Add error to intermediate steps
            state.add_intermediate_step(
                agent_name="research_agent",
                action="error",
                details={"error": error_details}
            )
            
            return state

# Function for use in the LangGraph workflow
def research_agent_node(state: AgentState) -> AgentState:
    """
    Node function for the research agent in the LangGraph workflow.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state after research agent processing
    """
    agent = ResearchAgent()
    return agent.process(state)