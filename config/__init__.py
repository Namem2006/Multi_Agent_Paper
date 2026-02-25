"""
LLM Configuration Package
Handles Azure OpenAI setup and guideline loading
"""

from .llm_config import get_llm, load_guideline, GUIDELINE_CONTENT

__all__ = ["get_llm", "load_guideline", "GUIDELINE_CONTENT"]
