"""
Tests for the Drafting Agent.
"""
import pytest
from unittest.mock import MagicMock, patch
from agents.drafting_agent import DraftingAgent
from models.state import AgentState
from langchain_core.messages import AIMessage

@pytest.fixture
def mock_llm():
    """Fixture to create a mock LLM."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="This is a synthesized answer based on the research results.")
    return mock

@pytest.fixture
def research_state():
    """Fixture to create a state with research results."""
    state = AgentState(
        query="What are the latest advancements in quantum computing?",
        research_results=[
            {
                "query": "quantum computing recent breakthroughs",
                "results": [
                    {
                        "title": "Recent Breakthrough in Quantum Computing",
                        "content": "Scientists have achieved a significant breakthrough in quantum computing...",
                        "url": "https://example.com/quantum-breakthrough"
                    }
                ]
            },
            {
                "query": "quantum supremacy experiments 2024",
                "results": [
                    {
                        "title": "Quantum Computing Applications in 2024",
                        "content": "The applications of quantum computing are expanding rapidly...",
                        "url": "https://example.com/quantum-applications"
                    }
                ]
            }
        ]
    )
    return state

def test_drafting_agent_initialization():
    """Test that the DraftingAgent initializes correctly."""
    agent = DraftingAgent()
    assert agent is not None
    assert agent.llm is not None
    assert agent.drafting_chain is not None

def test_drafting_agent_process_success(mock_llm, research_state):
    """Test the successful processing of research results."""
    # Setup
    agent = DraftingAgent(llm=mock_llm)
    
    # Execute
    result = agent.process(research_state)
    
    # Assert
    assert result.error is None
    assert result.final_answer is not None
    assert len(result.final_answer) > 0
    assert len(result.intermediate_steps) > 0
    assert "drafting_agent" in result.intermediate_steps[0]["agent"]
    
    # Verify chain was called with correct parameters
    mock_llm.invoke.assert_called_once()
    # Verify the prompt includes the research query
    args = mock_llm.invoke.call_args[0][0]
    assert research_state.query in str(args)

def test_drafting_agent_empty_research(mock_llm):
    """Test handling of empty research results."""
    # Setup
    agent = DraftingAgent(llm=mock_llm)
    empty_state = AgentState(
        query="What are the latest advancements in quantum computing?",
        research_results=[]
    )
    
    # Execute
    result = agent.process(empty_state)
    
    # Assert
    assert result.error is not None
    assert "No research results available" in result.error
    assert result.final_answer is None

def test_drafting_agent_with_partial_results(mock_llm):
    """Test processing with partial or limited research results."""
    # Setup
    agent = DraftingAgent(llm=mock_llm)
    partial_state = AgentState(
        query="What are the latest advancements in quantum computing?",
        research_results=[
            {
                "query": "quantum computing recent breakthroughs",
                "results": []  # Empty results for this sub-query
            }
        ]
    )
    
    # Execute
    result = agent.process(partial_state)
    
    # Assert
    assert result.error is None  # Should still process without error
    assert result.final_answer is not None
    assert "limited information" in result.final_answer.lower() or "insufficient data" in result.final_answer.lower()

@patch('agents.drafting_agent.DraftingAgent.extract_relevant_information')
def test_drafting_agent_information_extraction(mock_extract, mock_llm, research_state):
    """Test that the agent extracts relevant information from research results."""
    # Setup
    mock_extract.return_value = {
        "key_findings": ["Quantum computers now exceed 1000 qubits", "Error correction has improved"],
        "sources": ["https://example.com/quantum-breakthrough", "https://example.com/quantum-applications"]
    }
    agent = DraftingAgent(llm=mock_llm)
    
    # Execute
    result = agent.process(research_state)
    
    # Assert
    mock_extract.assert_called_once_with(research_state.research_results)
    assert result.error is None
    assert result.final_answer is not None

def test_drafting_agent_error_handling():
    """Test error handling during the drafting process."""
    # Setup
    failing_llm = MagicMock()
    failing_llm.invoke.side_effect = Exception("LLM processing error")
    agent = DraftingAgent(llm=failing_llm)
    state = AgentState(
        query="What are the latest advancements in quantum computing?",
        research_results=[{
            "query": "quantum computing",
            "results": [{"title": "Title", "content": "Content", "url": "https://example.com"}]
        }]
    )
    
    # Execute
    result = agent.process(state)
    
    # Assert
    assert result.error is not None
    assert "Error during drafting" in result.error
    assert result.final_answer is None

def test_drafting_agent_source_attribution(mock_llm, research_state):
    """Test that the final answer includes proper source attribution."""
    # Modify the mock to include source citations
    mock_llm.invoke.return_value = AIMessage(
        content="This is a synthesized answer based on the research results. [1][2]"
    )
    agent = DraftingAgent(llm=mock_llm)
    
    # Execute
    result = agent.process(research_state)
    
    # Assert
    assert result.error is None
    assert result.final_answer is not None
    
    # Check for source attribution in the answer or in the sources field
    source_urls = [
        "https://example.com/quantum-breakthrough",
        "https://example.com/quantum-applications"
    ]
    
    if hasattr(result, 'sources') and result.sources:
        # If sources are stored separately
        for url in source_urls:
            assert any(url in source for source in result.sources)
    else:
        # If sources are embedded in the answer
        answer_lower = result.final_answer.lower()
        assert "source" in answer_lower or "reference" in answer_lower or "[" in answer_lower