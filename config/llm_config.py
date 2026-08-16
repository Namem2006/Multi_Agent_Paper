"""Shared LLM configuration loaded from environment variables."""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT  = os.getenv("BASE_URL") or os.getenv("AZURE_ENDPOINT")
API_VERSION     = os.getenv("API_VERSION",    "2024-08-01-preview")
MODEL           = os.getenv("MODEL") or os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "3000"))

# ── Optional Google Gemini configuration ─────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_KEY")
GEMINI_MODEL_A1 = os.getenv("GEMINI_MODEL_A1", "gemini-2.5-flash")


def get_llm() -> AzureChatOpenAI:
    """Return the shared Azure OpenAI client used by the agent workflow."""
    if not OPENAI_API_KEY or not AZURE_ENDPOINT:
        raise ValueError(
            "Missing Azure OpenAI configuration. Set OPENAI_API_KEY and "
            "BASE_URL (or AZURE_ENDPOINT) in .env."
        )

    return AzureChatOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=API_VERSION,
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


def get_gemini_llm(model: str = None, temperature: float = 0.1):
    """Return an optional Google Gemini client."""
    if not GEMINI_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_KEY) in .env.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("langchain-google-genai is required. Run: pip install langchain-google-genai")
    return ChatGoogleGenerativeAI(
        model=model or GEMINI_MODEL_A1,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )


def load_guideline(guideline_path: str = None) -> str:
    """Load ACSA Guideline from file. Searches multiple candidate locations."""
    code_root = os.path.dirname(os.path.abspath(__file__))  # config/ dir
    project_root = os.path.dirname(code_root)               # code_for_github/

    candidates = []
    if guideline_path:
        candidates.append(guideline_path)

    env_path = os.getenv("GUIDELINE_PATH")
    if env_path:
        candidates.append(env_path)

    candidates += [
        os.path.join(project_root, "system_data", "adapted_guideline.txt"),
        os.path.join(project_root, "data", "guideline.txt"),
        os.path.join(project_root, "guideline.txt"),
        os.path.join(os.path.dirname(project_root), "guideline.txt"),
        "guideline.txt",
        "../guideline.txt",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                return content

    return (
        "Guideline file not found. "
        "Please place guideline.txt under code_for_github/data/ or run adapt_agent first."
    )


GUIDELINE_CONTENT = load_guideline()
