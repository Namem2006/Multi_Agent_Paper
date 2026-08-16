"""Token usage logger for all LLM calls (annotator, debate, judge, etc.)."""

import json
import os
from datetime import datetime, timezone

try:
    import tiktoken  # optional
except Exception:
    tiktoken = None


def _extract_token_usage(response_obj):
    if response_obj is None:
        return None, None, None, None

    # Try LangChain response metadata
    token_usage = None
    model_name = None

    if hasattr(response_obj, "response_metadata"):
        meta = response_obj.response_metadata or {}
        token_usage = meta.get("token_usage") or meta.get("usage")
        model_name = meta.get("model") or meta.get("model_name")

    if token_usage is None and hasattr(response_obj, "usage_metadata"):
        token_usage = response_obj.usage_metadata

    if token_usage is None and hasattr(response_obj, "additional_kwargs"):
        token_usage = response_obj.additional_kwargs.get("token_usage")

    if isinstance(token_usage, dict):
        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")
        total_tokens = token_usage.get("total_tokens")
        return prompt_tokens, completion_tokens, total_tokens, model_name

    return None, None, None, model_name


def _estimate_tokens(text, model_name=None):
    if text is None:
        return 0

    if tiktoken is None:
        # Fallback heuristic: ~4 chars per token
        return max(1, int(len(text) / 4)) if text else 0

    try:
        if model_name:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    return len(enc.encode(text))


def _load_totals(totals_file):
    if os.path.exists(totals_file):
        try:
            with open(totals_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "session_started_utc": datetime.now(timezone.utc).isoformat(),
        "overall": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "agents": {},
    }


def _save_totals(totals_file, totals):
    with open(totals_file, "w", encoding="utf-8") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)


def log_llm_usage(agent_name, input_text, output_text, response_obj=None, model_name=None, call_type=None, log_dir=None):
    if not log_dir:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "system_data")

    os.makedirs(log_dir, exist_ok=True)

    prompt_tokens, completion_tokens, total_tokens, model_from_resp = _extract_token_usage(response_obj)
    if model_name is None:
        model_name = model_from_resp

    estimated = False
    if prompt_tokens is None or completion_tokens is None:
        prompt_tokens = _estimate_tokens(input_text, model_name)
        completion_tokens = _estimate_tokens(output_text, model_name)
        total_tokens = prompt_tokens + completion_tokens
        estimated = True

    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "call_type": call_type,
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated": estimated,
    }

    log_file = os.path.join(log_dir, "llm_token_usage.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    totals_file = os.path.join(log_dir, "llm_token_usage_totals.json")
    totals = _load_totals(totals_file)

    totals["overall"]["prompt_tokens"] += prompt_tokens
    totals["overall"]["completion_tokens"] += completion_tokens
    totals["overall"]["total_tokens"] += total_tokens

    agent_bucket = totals["agents"].setdefault(agent_name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    agent_bucket["prompt_tokens"] += prompt_tokens
    agent_bucket["completion_tokens"] += completion_tokens
    agent_bucket["total_tokens"] += total_tokens

    _save_totals(totals_file, totals)


def reset_token_usage(log_dir=None):
    if not log_dir:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "system_data")

    os.makedirs(log_dir, exist_ok=True)

    totals_file = os.path.join(log_dir, "llm_token_usage_totals.json")
    log_file = os.path.join(log_dir, "llm_token_usage.jsonl")

    totals = {
        "session_started_utc": datetime.now(timezone.utc).isoformat(),
        "overall": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "agents": {},
    }
    _save_totals(totals_file, totals)

    # Truncate per-call log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")
