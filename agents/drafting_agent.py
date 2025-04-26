"""
Drafting Agent implementation for the AI Agentic Research System.
Responsible for synthesizing research results into a coherent answer.
"""
from typing import List, Dict, Any
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models.state import AgentState
from config.settings import get_settings
from agents.utils import format_error, truncate_text

# Get logger
logger = logging.getLogger(__name__)

class DraftingAgent:
    """
    Agent responsible for synthesizing research results into a coherent answer.
    """
    def __init__(self, llm=None):
        settings = get_settings()
        
        # Initialize LLM
        self.llm = llm or ChatOpenAI(
            model=settings.default_model,
            temperature=settings.drafting_agent_temperature
        )
        
        # Setup the drafting prompt
        self.drafting_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert researcher and writer. Based on the original query and the research results,
            create a comprehensive, well-structured response. 
            
            ORIGINAL QUERY: {query}
            
            RESEARCH RESULTS:
            {research_results}
            
            Your task is to:
            1. Synthesize the information from all sources
            2. Organize it logically with clear section headings where appropriate
            3. Present a clear, comprehensive answer to the original query
            4. Cite specific sources where appropriate using [Source X] notation
            5. Include a "Sources" section at the end that lists all the URLs used
            
            Provide a thoughtful, nuanced response that thoroughly addresses the query.
            If the research results are insufficient to properly answer the query, acknowledge this limitation
            and suggest what additional information would be helpful.
            """
        )
        
        # Create the drafting chain
        self.drafting_chain = self.drafting_prompt | self.llm
    
    def process(self, state: AgentState) -> AgentState:
        """
        Process the research results and draft a comprehensive answer.
        
        Args:
            state: Current state of the agent system with research results
            
        Returns:
            Updated state with final answer
        """
        logger.info("Drafting agent processing research results")
        settings = get_settings()
        
        try:
            # Check if we have research results
            if not state.research_results:
                logger.warning("No research results to process")
                state.final_answer = "Unable to generate an answer as no research results were collected."
                return state
            
            # Extract all search results
            all_results = []
            for search_group in state.research_results:
                for result in search_group["results"]:
                    all_results.append({
                        "title": result.get("title", "No title"),
                        "content": result.get("content", "No content"),
                        "url": result.get("url", "No URL")
                    })
            
            # Truncate if too many results to fit context window
            max_results = settings.max_drafting_sources
            if len(all_results) > max_results:
                logger.info(f"Truncating results from {len(all_results)} to {max_results}")
                all_results = all_results[:max_results]
                
            # Convert results to formatted text
            formatted_results = ""
            for i, result in enumerate(all_results):
                # Truncate content if too long
                content = truncate_text(
                    result["content"], 
                    max_length=settings.max_source_content_length
                )
                
                formatted_results += f"Source {i+1}:\nTitle: {result['title']}\nContent: {content}\nURL: {result['url']}\n\n"
            
            # Generate comprehensive answer
            logger.info("Generating final answer")
            response = self.drafting_chain.invoke({
                "query": state.query,
                "research_results": formatted_results
            })
            
            final_answer = response.content
            logger.info(f"Final answer generated, length: {len(final_answer)} characters")
            
            # Update state with final answer
            state.final_answer = final_answer
            
            # Add intermediate step
            state.add_intermediate_step(
                agent_name="drafting_agent",
                action="synthesize",
                details={
                    "sources_used": len(all_results),
                    "answer_length": len(final_answer)
                }
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Error in drafting agent: {str(e)}")
            error_details = format_error(e)
            
            # Update state with error
            state.error = f"Drafting agent error: {error_details}"
            
            # Add error to intermediate steps
            state.add_intermediate_step(
                agent_name="drafting_agent",
                action="error",
                details={"error": error_details}
            )
            
            return state

# Function for use in the LangGraph workflow
def drafting_agent_node(state: AgentState) -> AgentState:
    """
    Node function for the drafting agent in the LangGraph workflow.
    
    Args:
        state: Current state of the workflow with research results
        
    Returns:
        Updated state after drafting agent processing
    """
    agent = DraftingAgent()
    return agent.process(state)