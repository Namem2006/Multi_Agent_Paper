import os
import json
import re
import time
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from models.schemas import DebateState
from memory_and_history.history_manager import HistoryManager
from config.llm_config import get_llm
from prompts.debate_prompts import create_summary_prompt_template
from utils.llm_invoke_with_timeout import invoke_with_timeout_and_retry

def _strip_code_fences(output: str) -> str:
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def clean_json_output(output: str) -> Optional[Dict[str, Any]]:
    if not output:
        return None

    text = _strip_code_fences(output)
    candidates = [text]

    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    return None

class SummaryAgent:
    def __init__(self, llm=None):
        self.history_manager = HistoryManager()
        self.llm = llm if llm is not None else get_llm()
        self.summary_prompt_template = create_summary_prompt_template()

    def summarize_response(self, full_response: Dict[str, Any], agent_name: str, opponent_name: str, round_number: int) -> Dict[str, Any]:
        num_conflicts = len(full_response.get("labels", []))
        full_response_json = json.dumps(full_response, ensure_ascii=False, indent=2)

        prompt = self.summary_prompt_template.format(
            full_response=full_response_json,
            agent_name=agent_name,
            opponent_name=opponent_name,
            round_number=round_number,
            num_conflicts=num_conflicts
        )

        parsed_resp = None
        last_error = ""
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            attempt_prompt = prompt
            if attempt > 1:
                attempt_prompt += (
                    "\n\nIMPORTANT: Return ONLY a valid JSON object with keys labels, opinion, evidence. "
                    "No markdown and no extra text."
                )

            messages = [HumanMessage(content=attempt_prompt)]

            try:
                # Tránh timeout khi API chậm hoặc mạng lag
                raw_response = invoke_with_timeout_and_retry(
                    self.llm,
                    messages,
                    agent_name="Summary Agent",
                    call_type="summary_agent",
                )
                if raw_response is None:
                    last_error = "LLM invoke failed after retries"
                else:
                    raw_content = raw_response.content if hasattr(raw_response, "content") else ""
                    if not isinstance(raw_content, str):
                        raw_content = str(raw_content)

                    parsed_resp = clean_json_output(raw_content)
                    if parsed_resp and isinstance(parsed_resp.get("opinion", ""), str) and isinstance(parsed_resp.get("evidence", ""), str):
                        break

                    last_error = f"Malformed JSON output on attempt {attempt}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < max_attempts:
                wait_s = 0.8 * attempt
                print(f"    [WARN] Summary retry {attempt}/{max_attempts} due to: {last_error}")
                time.sleep(wait_s)

        if not parsed_resp:
            short_error = last_error[:180] if last_error else "Unknown parsing error"
            fallback_opinion = full_response.get("opinion", "")
            if not isinstance(fallback_opinion, str) or not fallback_opinion.strip():
                fallback_opinion = (
                    f"Countering {opponent_name}'s argument in round {round_number}, "
                    f"a fallback summary was used because structured JSON parsing failed ({short_error})."
                )

            fallback_evidence = full_response.get("evidence", "")
            if not isinstance(fallback_evidence, str) or not fallback_evidence.strip():
                fallback_evidence = "Fallback: no valid summary evidence returned by model."

            parsed_resp = {
                "opinion": fallback_opinion,
                "evidence": fallback_evidence,
            }

        required_prefix = f"Countering {opponent_name}'s argument in round {round_number},"
        opinion_text = parsed_resp.get("opinion", "")
        if isinstance(opinion_text, str) and opinion_text.strip() and not opinion_text.strip().startswith(required_prefix):
            opinion_text = f"{required_prefix} {opinion_text.strip()}"
        elif not isinstance(opinion_text, str) or not opinion_text.strip():
            opinion_text = f"{required_prefix} summary unavailable due to malformed model output."

        summarized = {
            "labels": full_response.get("labels", []),
            "opinion": opinion_text,
            "evidence": parsed_resp.get("evidence", "")
        }
        return summarized

    def record_response(self, state: DebateState) -> DebateState:
        last_response = state.get("last_response", {})
        annotator = last_response.get("annotator")
        full_response = last_response.get("response")

        if not annotator or not full_response:
            return state

        current_case = state["current_case"]
        current_round = self.history_manager.get_current_round(state)
        print(f"  → Moderator đang tóm tắt và ghi nhận response của {annotator} (Case {current_case}, Round {current_round})...")

        opponent_name = "A2" if annotator == "A1" else "A1"
        opponent_last_round = 0 if current_round == 1 else current_round - 1

        summarized_response = self.summarize_response(full_response, annotator, opponent_name, opponent_last_round)
        state = self.history_manager.append_response(state, annotator, summarized_response)
        state["current_turn"] = None
        return state

def create_summary_agent(llm=None) -> SummaryAgent:
    return SummaryAgent(llm=llm)