"""
LLM Configuration and Guideline Loading

Handles:
- Azure OpenAI LLM initialization
- Guideline loading from file
- Environment configuration
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()


# Azure OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("AZURE_ENDPOINT", "")
API_VERSION = os.getenv("API_VERSION", "")
MODEL = os.getenv("MODEL", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))


def get_llm() -> AzureChatOpenAI:
    """
    Initialize and return Azure OpenAI LLM instance
    
    Returns:
        Configured AzureChatOpenAI instance
    """
    return AzureChatOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=BASE_URL,
        api_version=API_VERSION,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )


def load_guideline(guideline_path: str = None) -> str:
    """
    Load ACSA Guideline from file
    
    Args:
        guideline_path: Path to guideline file (default: search in multiple locations)
        
    Returns:
        Guideline content as string
    """
    if guideline_path is None:
        # Try multiple locations
        possible_paths = [
            "guideline.txt",  # Current directory
            "../guideline.txt",  # Parent directory
            os.path.join(os.path.dirname(__file__), "../guideline.txt"),
            os.path.join(os.path.dirname(__file__), "../../guideline.txt"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                guideline_path = path
                break
    
    try:
        with open(guideline_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Guideline file '{guideline_path}' not found.")
        print("Trying alternative paths...")
        return "Guideline file not found. Please set GUIDELINE_PATH in .env or place guideline.txt in the project root."


# Load guideline once at module import
GUIDELINE_CONTENT = load_guideline()
