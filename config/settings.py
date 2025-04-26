"""
Settings for the AI Agentic Research System.
"""
import os
import logging
from typing import Dict, Any
from functools import lru_cache
from langchain_core.pydantic_v1 import BaseSettings

class Settings(BaseSettings):
    """
    Settings for the AI Agentic Research System.
    Loads from environment variables.
    """
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # LLM Settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4")
    research_agent_temperature: float = 0.1
    drafting_agent_temperature: float = 0.2
    
    # Research Agent Settings
    num_search_queries: int = 3
    max_search_results_per_query: int = 5
    
    # Drafting Agent Settings
    max_drafting_sources: int = 15
    max_source_content_length: int = 500
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings. Results are cached.
    
    Returns:
        Application settings
    """
    return Settings()

def setup_logging() -> None:
    """
    Set up logging for the application.
    """
    settings = get_settings()
    
    # Convert string log level to numeric
    numeric_level = getattr(logging, settings.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=settings.log_format
    )
    
    # Set specific levels for third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)