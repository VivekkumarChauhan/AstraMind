# AI Agentic Research System

A multi-agent system for deep research that crawls websites using Tavily for online information gathering, utilizing LangGraph & LangChain frameworks for efficient information organization and processing.

## Overview

This project implements an AI agentic system designed for deep research tasks. The system consists of two primary agents:

1. **Research Agent**: Responsible for gathering information from the web using the Tavily search API
2. **Drafting Agent**: Synthesizes the collected information into comprehensive, well-structured answers

The system leverages LangGraph for orchestrating the workflow between these agents and LangChain for building the agent components.

## Architecture

The system follows a directed graph architecture using LangGraph:

```
[User Query] → [Research Agent] → [Drafting Agent] → [Final Answer]
```

- **State Management**: The system uses a shared state model (`AgentState`) to pass information between agents
- **Tool Integration**: Incorporates Tavily search for real-time web information retrieval
- **Workflow Control**: Uses conditional routing to determine the execution flow

## Features

- **Dynamic Query Refinement**: Intelligently generates targeted search queries based on the user's initial query
- **Multi-Source Research**: Collects and processes information from multiple web sources
- **Source Citation**: Properly cites sources in the final output for attribution
- **Error Handling**: Robust error handling throughout the workflow
- **Persistent Sessions**: Maintains conversation context through session management

## Installation

1. Clone this repository:
```bash
git clone https://github.com/VivekkumarChauhan/AstraMind.git
cd AstraMind
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### Basic Usage

```python
from research_system import ResearchSystem

# Initialize the system
system = ResearchSystem()
system.start_session()

# Process a query
result = system.process_query("What are the latest advancements in quantum computing?")

# Display the result
print(result["answer"])
```

### Advanced Usage

```python
# Start a new session with a specific LLM model
system = ResearchSystem(llm_model="gpt-4-turbo")
session_id = system.start_session()

# Process multiple queries in the same session
result1 = system.process_query("Explain the impact of AI on healthcare")
result2 = system.process_query("How is AI being used for drug discovery?")

# Get detailed statistics about the research process
print(f"Research queries: {result2['research_queries']}")
print(f"Sources used: {result2['sources_count']}")
```

## Project Structure

```
project/
├── README.md                # Project documentation
├── .env                     # Environment variables (API keys)
├── requirements.txt         # Project dependencies
├── main.py                  # Main entry point
├── agents/                  # Agent implementations
│   ├── __init__.py
│   ├── research_agent.py    # Research agent logic
│   ├── drafting_agent.py    # Drafting agent logic
│   └── utils.py             # Shared utility functions
├── models/                  # Data models
│   ├── __init__.py
│   └── state.py             # Agent state definition
├── config/                  # Configuration
│   ├── __init__.py
│   └── settings.py          # System settings
└── tests/                   # Test suite
    ├── __init__.py
    ├── test_research_agent.py
    └── test_drafting_agent.py
```

## Implementation Details

### Research Agent

The research agent is responsible for:

1. Analyzing the user query to identify key information needs
2. Generating targeted search queries to gather relevant information
3. Executing searches using the Tavily API
4. Processing and storing search results in the shared state

### Drafting Agent

The drafting agent is responsible for:

1. Processing all gathered research data
2. Synthesizing information from multiple sources
3. Organizing content in a logical structure
4. Creating a comprehensive answer with source citations
5. Delivering a polished final response

### LangGraph Workflow

The workflow manages the interaction between agents:

1. Starts with the Research Agent to gather information
2. Routes to the Drafting Agent once research is complete
3. Delivers the final answer when drafting is complete
4. Handles errors gracefully throughout the process

## Dependencies

- **LangChain**: Framework for building LLM applications
- **LangGraph**: Framework for creating agent workflows
- **Tavily API**: Web search and information retrieval
- **OpenAI API**: Language model for agents

## Future Improvements

- Add a fact-checking agent to verify information
- Implement user feedback loops for improved answers
- Add support for multimedia content sources
- Enhance the query refinement process with user context
- Implement caching for improved performance

## About the Developer

Vivekkumar Chauhan - Passionate about AI, natural language processing, and building systems that can effectively gather and synthesize information. Experienced in Python programming with a strong background in machine learning and AI frameworks.
