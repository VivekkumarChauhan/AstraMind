"""
Tests for the Research Agent.
"""
import pytest
from unittest.mock import MagicMock, patch
from agents.research_agent import ResearchAgent
from models.state import AgentState

@pytest.fixture
def mock_llm():
    """Fixture to create a mock LLM."""
    mock = MagicMock()
    mock.invoke.return_value = {
        "search_queries": [
            "quantum computing recent breakthroughs",
            "quantum supremacy experiments 2024",
            "quantum computing applications"
        ],
        "reasoning": "These queries cover recent advancements, experimental results, and practical applications."
    }
    return mock

@pytest.fixture
def mock_tool_executor():
    """Fixture to create a mock tool executor."""
    mock = MagicMock()
    mock.invoke.return_value = [
        {
            "title": "Recent Breakthrough in Quantum Computing",
            "content": "Scientists have achieved a significant breakthrough in quantum computing...",
            "url": "https://example.com/quantum-breakthrough"
        },
        {
            "title": "Quantum Computing Applications in 2024",
            "content": "The applications of quantum computing are expanding rapidly...",
            "url": "https://example.com/quantum-applications"
        }
    ]
    return mock

def test_research_agent_initialization():
    """Test that the ResearchAgent initializes correctly."""
    agent = ResearchAgent()
    assert agent is not None
    assert agent.llm is not None
    assert agent.tools is not None
    assert agent.tool_executor is not None

@patch("agents.research_agent.JsonOutputParser")
def test_research_agent_process_success(mock_parser, mock_llm, mock_tool_executor):
    """Test the successful processing of a query."""
    # Setup
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "search_queries": [
            "quantum computing recent breakthroughs",
            "quantum supremacy experiments 2024"
        ],
        "reasoning": "These queries target the most recent developments."
    }
    
    agent = ResearchAgent(llm=mock_llm)
    agent.query_generation_chain = mock_chain
    agent.tool_executor = mock_tool_executor
    
    # Create initial state
    state = AgentState(query="What are the latest advancements in quantum computing?")
    
    # Execute
    result = agent.process(state)
    
    # Assert
    assert result.error is None
    assert len(result.research_results) > 0
    assert len(result.intermediate_steps) > 0
    assert "research_agent" in result.intermediate_steps[0]["agent"]
    
    # Verify chain was called with correct parameters
    mock_chain.invoke.assert_called_once()
    assert "query" in mock_chain.invoke.call_args[0][0]

def test_research_agent_process_error(mock_llm):
    """Test handling of errors during processing."""
    # Setup
    agent = ResearchAgent(llm=mock_llm)
    agent.query_generation_chain = MagicMock()
    agent.query_generation_chain.invoke.side_effect = Exception("Test error")
    
    # Create initial state
    state = AgentState(query="What are the latest advancements in quantum computing?")
    
    # Execute
    result = agent.process(state)
    
    # Assert
    assert result.error is not None
    assert "Test error" in result.error
    assert len(result.research_results) == 0