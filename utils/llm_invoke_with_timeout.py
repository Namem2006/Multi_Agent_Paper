"""
LLM invoke helper with timeout protection for all agents (Debate, Judge, etc.)
Shared by all pipeline stages to prevent terminal freeze during API calls.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ThreadTimeoutError
from dotenv import load_dotenv
import os
from utils.token_usage_logger import log_llm_usage

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

LLM_INVOKE_TIMEOUT_SEC = float(os.getenv("LLM_INVOKE_TIMEOUT_SEC", "90.0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "10"))
LLM_BASE_SLEEP = float(os.getenv("LLM_BASE_SLEEP", "1.0"))

def _messages_to_text(messages):
    parts = []
    for msg in messages or []:
        if isinstance(msg, dict):
            parts.append(str(msg.get("content", "")))
        else:
            content = getattr(msg, "content", "")
            parts.append(str(content))
    return "\n\n".join(parts)



def invoke_with_timeout_and_retry(
    llm,
    messages,
    agent_name: str = "Agent",
    timeout_sec: float = None,
    call_type: str = None,
):
    """
    Invoke LLM with timeout protection and automatic retry.

    Args:
        llm: LangChain LLM instance (AzureChatOpenAI, etc.)
        messages: Messages to send to LLM (List[Dict] or List[HumanMessage])
        agent_name: Name of calling agent for logging
        timeout_sec: Timeout in seconds (default from .env)

    Returns:
        LLM response object or None if all retries failed
    """
    if timeout_sec is None:
        timeout_sec = LLM_INVOKE_TIMEOUT_SEC

    last_error = "Unknown error"

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            print(f"[{agent_name}] Attempt {attempt}/{LLM_MAX_RETRIES} (timeout={timeout_sec:.0f}s)...", flush=True)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm.invoke, messages)
                response = future.result(timeout=timeout_sec)

            print(f"[{agent_name}] Success on attempt {attempt}.", flush=True)
            input_text = _messages_to_text(messages)
            output_text = getattr(response, "content", "") if response is not None else ""
            log_call_type = call_type if call_type else agent_name
            log_llm_usage(agent_name, input_text, str(output_text), response_obj=response, call_type=log_call_type)
            return response

        except ThreadTimeoutError:
            last_error = f"Timeout after {timeout_sec:.0f}s (API slow or network lag)"
            print(f"[{agent_name}] {last_error}", flush=True)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[{agent_name}] Exception: {last_error}", flush=True)

        if attempt < LLM_MAX_RETRIES:
            wait_s = LLM_BASE_SLEEP * attempt
            print(
                f"[{agent_name}] Retry {attempt}/{LLM_MAX_RETRIES} after {wait_s:.1f}s...",
                flush=True
            )
            time.sleep(wait_s)

    print(f"[{agent_name}] Failed after {LLM_MAX_RETRIES} attempts. Last error: {last_error}", flush=True)
    return None
