"""
LangGraph workflow definition for the AI Agentic Research System.
"""
from langgraph.graph import StateGraph, END
from models.state import AgentState
from agents.research_agent import research_agent_node
from agents.drafting_agent import drafting_agent_node
import logging

logger = logging.getLogger(__name__)

def router(state: AgentState) -> str:
    """
    Routes the flow between agents or to completion.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Next node name or END
    """
    # Check for errors first
    if state.error:
        logger.warning(f"Workflow encountered an error: {state.error}")
        return END
    
    # Route based on workflow progress
    if not state.research_results:
        logger.info("Routing to research agent")
        return "research"
    elif not state.final_answer:
        logger.info("Routing to drafting agent")
        return "draft"
    else:
        logger.info("Workflow complete")
        return END

def create_workflow():
    """
    Create and return the agent workflow graph.
    
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Creating agent workflow")
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_agent_node)
    workflow.add_node("draft", drafting_agent_node)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add edges with routing logic
    workflow.add_conditional_edges("research", router)
    workflow.add_conditional_edges("draft", router)
    
    # Compile the graph
    logger.info("Agent workflow created and compiled")
    return workflow.compile()