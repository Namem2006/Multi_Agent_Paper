import os
import json
import re
import time
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI

from models.schemas import DebateState
from config.llm_config import get_llm
from prompts.debate_prompts import create_debate_prompt_template
from memory_and_history.history_manager import HistoryManager
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

    # Common case: model adds prose before/after JSON.
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            # Light repair for trailing commas.
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

    return None

class DebateAgent:
    def __init__(self, agent_name: str, llm: AzureChatOpenAI = None):
        self.agent_name = agent_name
        self.llm = llm if llm else get_llm()
        self.prompt_template = create_debate_prompt_template()
        self.history_manager = HistoryManager()

    def generate_response(self, state: DebateState) -> DebateState:
        current_case = state["current_case"]
        current_round = self.history_manager.get_current_round(state)
        my_initial = state["A1_initial"] if self.agent_name == "A1" else state["A2_initial"]
        my_labels = my_initial.get("labels", [])

        print(f"  → {self.agent_name} đang tranh luận (Case {current_case}, Round {current_round})...")

        prompt = self._build_prompt(state)

        parsed_resp = None
        last_error = ""
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                attempt_prompt = prompt
                if attempt > 1:
                    attempt_prompt += (
                        "\n\nIMPORTANT: Return ONLY one valid JSON object with keys "
                        "labels, opinion, evidence. No markdown, no explanation outside JSON."
                    )

                # Bỏ with_structured_output để chống lỗi Azure 400 Bad Request
                raw_response = invoke_with_timeout_and_retry(
                    self.llm,
                    [{"role": "user", "content": attempt_prompt}],
                    agent_name=self.agent_name
                )

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
                print(f"    [WARN] {self.agent_name} retry {attempt}/{max_attempts} due to: {last_error}")
                time.sleep(wait_s)

        if not parsed_resp:
            short_error = last_error[:180] if last_error else "Unknown parsing error"
            parsed_resp = {
                "opinion": (
                    f"Countering {'A2' if self.agent_name == 'A1' else 'A1'}'s argument in round {max(current_round - 1, 0)}, "
                    f"a fallback response was used because structured JSON parsing failed ({short_error})."
                ),
                "evidence": "Fallback: no valid structured evidence returned by the model.",
            }

        state["last_response"] = {
            "annotator": self.agent_name,
            "response": {
                # Debate labels must remain unchanged from initial position.
                "labels": my_labels,
                "opinion": parsed_resp.get("opinion", ""),
                "evidence": parsed_resp.get("evidence", "")
            }
        }
        state["current_turn"] = "moderator"

        return state

    def _build_prompt(self, state: DebateState) -> str:
        opponent = "A2" if self.agent_name == "A1" else "A1"

        my_initial  = state["A1_initial"] if self.agent_name == "A1" else state["A2_initial"]
        opp_initial = state["A2_initial"] if self.agent_name == "A1" else state["A1_initial"]

        my_labels  = my_initial.get("labels", [])
        opp_labels = opp_initial.get("labels", [])

        my_labels_list = []
        for i, lbl in enumerate(my_labels, start=1):
            my_labels_list.append(f"  {i}. {lbl.get('entity','')}#{lbl.get('attribute','')} | {lbl.get('sentiment','')}")
        my_labels_text = "\n".join(my_labels_list)

        opp_labels_list = []
        for i, lbl in enumerate(opp_labels, start=1):
            opp_labels_list.append(f"  {i}. {lbl.get('entity','')}#{lbl.get('attribute','')} | {lbl.get('sentiment','')}")
        opponent_labels_text = "\n".join(opp_labels_list)

        my_map  = {(l.get("entity"), l.get("attribute")): l.get("sentiment") for l in my_labels}
        opp_map = {(l.get("entity"), l.get("attribute")): l.get("sentiment") for l in opp_labels}

        diff_lines = []
        for (ent, attr), my_sent in my_map.items():
            opp_sent = opp_map.get((ent, attr))
            if opp_sent is None:
                diff_lines.append(f"  - {ent}#{attr}: I={my_sent}, opponent did NOT assign this label")
            elif opp_sent != my_sent:
                diff_lines.append(f"  - {ent}#{attr}: my_label={my_sent} vs opponent={opp_sent}")
        for (ent, attr), opp_sent in opp_map.items():
            if (ent, attr) not in my_map:
                diff_lines.append(f"  - {ent}#{attr}: I did NOT assign, opponent={opp_sent}")
        differing_summary = "\n".join(diff_lines) if diff_lines else "  (No differing labels — both annotators fully agree)"

        my_labels_json_block = json.dumps(
            [{"entity": l.get("entity"), "attribute": l.get("attribute"), "sentiment": l.get("sentiment")} for l in my_labels],
            ensure_ascii=False, indent=2
        )

        hint_lines = []
        for (ent, attr), my_sent in my_map.items():
            opp_sent = opp_map.get((ent, attr))
            if opp_sent is None:
                hint_lines.append(f"  For {ent}#{attr}|{my_sent}: Cite the Guideline rule that mandates this label.")
            elif opp_sent != my_sent:
                hint_lines.append(f"  For {ent}#{attr} (mine={my_sent}, opp={opp_sent}): Cite the Guideline rule.")
        for (ent, attr), opp_sent in opp_map.items():
            if (ent, attr) not in my_map:
                hint_lines.append(f"  For {ent}#{attr}|{opp_sent} (opp assigned, I did not): Cite the Guideline rule.")
        if not hint_lines:
            hint_lines = ["  (No differing labels — provide general accuracy analysis)"]
        differing_labels_hints = "\n".join(hint_lines)

        history_text, opponent_last_round = self.history_manager.build_history_text(state, self.agent_name)

        # ĐỌC RAG TỪ STATE (được truyền bởi workflow_controller)
        retrieved_context = state.get("retrieved_context", "(No specific guideline context retrieved.)")

        prompt = self.prompt_template.format(
            review_text=state["text"],
            retrieved_context=retrieved_context,
            my_name=self.agent_name,
            my_labels_text=my_labels_text,
            opponent_name=opponent,
            opponent_labels_text=opponent_labels_text,
            differing_summary=differing_summary,
            conversation_history=history_text,
            opponent_last_round=opponent_last_round,
            my_labels_json_block=my_labels_json_block,
            differing_labels_hints=differing_labels_hints,
            num_my_labels=len(my_labels),
        )

        return prompt

def create_debate_agents(llm=None) -> Dict[str, DebateAgent]:
    return {
        "A1": DebateAgent("A1", llm=llm),
        "A2": DebateAgent("A2", llm=llm)
    }