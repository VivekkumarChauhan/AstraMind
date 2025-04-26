"""
Utility functions for agents in the AI Agentic Research System.
"""
import traceback
from typing import Dict, Any

def format_error(exception: Exception) -> str:
    """
    Format an exception into a more readable error message.
    
    Args:
        exception: The exception to format
        
    Returns:
        Formatted error message
    """
    error_type = type(exception).__name__
    error_msg = str(exception)
    
    # Get traceback info but limit it to avoid overwhelming responses
    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    tb_text = "".join(tb_lines[-3:])  # Only include the last 3 lines
    
    return f"{error_type}: {error_msg}\n{tb_text}"

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length while maintaining sentence integrity.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the truncated text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    
    # Find the last sentence boundary
    for ending in [". ", "! ", "? "]:
        last_period = truncated.rfind(ending)
        if last_period != -1:
            return truncated[:last_period + 1] + "..."
    
    # If no sentence boundary found, truncate at word boundary
    last_space = truncated.rfind(" ")
    if last_space != -1:
        return truncated[:last_space] + "..."
    
    # If no word boundary found, just truncate
    return truncated + "..."

def extract_urls_from_results(results: Dict[str, Any]) -> list:
    """
    Extract all URLs from search results.
    
    Args:
        results: Search results from the research agent
        
    Returns:
        List of unique URLs
    """
    urls = set()
    
    for search_group in results:
        for result in search_group.get("results", []):
            if "url" in result:
                urls.add(result["url"])
    
    return list(urls)
# Add this to agents/utils.py
class SimpleToolExecutor:
    """A simplified tool executor if the langchain import doesn't work."""
    
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
    
    def invoke(self, tool_invocation):
        tool_name = tool_invocation["name"]
        tool_input = tool_invocation["input"]
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        tool = self.tools[tool_name]
        return tool.invoke(tool_input)