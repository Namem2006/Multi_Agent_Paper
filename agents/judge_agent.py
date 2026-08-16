import os
import json
import math
import random
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from config.llm_config import get_llm
from prompts.judge_prompts import (
    create_judge_prompt_template_a1_first,
    create_judge_prompt_template_a2_first,
    create_judge_tiebreak_prompt_template,
)
from utils.llm_invoke_with_timeout import invoke_with_timeout_and_retry

def clean_json_output(output: str):
    output = output.strip()
    if output.startswith("```json"):
        output = output.replace("```json\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]
    elif output.startswith("```"):
        output = output.replace("```\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {
            "decision": {
                "winner_annotator": "A1",
                "confidence": 0.50,
                "reasoning": "JSON Error",
                "key_evidence": "",
            }
        }


def _normalize_winner(raw_winner: Any) -> str:
    if isinstance(raw_winner, str):
        w = raw_winner.strip().upper()
        if w in ["A1", "A2"]:
            return w
    return "A1"


def _normalize_confidence(raw_confidence: Any) -> float:
    try:
        c = float(raw_confidence)
    except (TypeError, ValueError):
        c = 0.50

    if math.isnan(c) or math.isinf(c):
        c = 0.50

    c = max(0.0, min(1.0, c))
    return round(c, 2)


def _normalize_confidence_level(raw_level: Any, confidence: float) -> str:
    if isinstance(raw_level, str):
        level = raw_level.strip().upper()
        if level in ["HIGH", "MEDIUM", "LOW"]:
            return level

    if confidence > 0.90:
        return "HIGH"
    if confidence >= 0.60:
        return "MEDIUM"
    return "LOW"

class JudgeAgent:
    def __init__(self, judge_id: int, llm=None, view_mode: str = "A1_FIRST"):
        self.judge_id = judge_id
        self.judge_name = f"Judge_{judge_id}"
        self.llm = llm if llm is not None else get_llm()
        self.view_mode = view_mode if view_mode in ["A1_FIRST", "A2_FIRST"] else "A1_FIRST"

        if self.view_mode == "A2_FIRST":
            self.prompt_template = create_judge_prompt_template_a2_first()
        else:
            self.prompt_template = create_judge_prompt_template_a1_first()

    def _format_history(self, history: List[Dict[str, Any]], speaker_order: List[str]) -> str:
        lines = []
        for entry in history:
            round_num = entry.get("round", "?")
            for agent_name in speaker_order:
                if agent_name in entry:
                    data = entry[agent_name]
                    opinion = data.get("opinion", "")
                    evidence = data.get("evidence", "")
                    if "labels" in data:
                        labels_str = ", ".join(f"{l.get('entity','')}#{l.get('attribute','')}|{l.get('sentiment','')}" for l in data["labels"])
                    else:
                        labels_str = "(unknown)"
                    block = (
                        f"[Round {round_num} - {agent_name}]\n"
                        f"  Labels: {labels_str}\n"
                        f"  Opinion: {opinion}\n"
                        f"  Evidence: {evidence}"
                    )
                    lines.append(block)
        return "\n\n".join(lines) if lines else "(No debate history)"

    def _format_labels_text(self, labels: List[Dict[str, Any]]) -> str:
        lines = []
        for i, lbl in enumerate(labels, start=1):
            lines.append(f"  {i}. {lbl.get('entity','')}#{lbl.get('attribute','')} | {lbl.get('sentiment','')}")
        return "\n".join(lines) if lines else "  (No labels)"

    def decide(self, debate_result: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
        review_text = debate_result.get("review_text", "")
        initial_positions = debate_result.get("initial_positions", {})

        a1_labels_text = self._format_labels_text(initial_positions.get("A1", {}).get("labels", []))
        a2_labels_text = self._format_labels_text(initial_positions.get("A2", {}).get("labels", []))

        case_1_history = self._format_history(
            debate_result.get("debate_summary", {}).get("case_1", {}).get("history", []),
            speaker_order=["A1", "A2"],
        )
        case_2_history = self._format_history(
            debate_result.get("debate_summary", {}).get("case_2", {}).get("history", []),
            speaker_order=["A2", "A1"],
        )

        # Dùng RAG context được truyền vào
        if not retrieved_context:
            retrieved_context = "(No specific guideline context retrieved.)"

        prompt = self.prompt_template.format(
            judge_name=self.judge_name,
            review_text=review_text,
            a1_labels_text=a1_labels_text,
            a2_labels_text=a2_labels_text,
            retrieved_context=retrieved_context,
            case_1_history=case_1_history,
            case_2_history=case_2_history,
        )

        messages = [HumanMessage(content=prompt)]

        raw_response = invoke_with_timeout_and_retry(
            self.llm,
            messages,
            agent_name=self.judge_name
        )
        parsed_resp = clean_json_output(raw_response.content)

        d = parsed_resp.get("decision", {})
        winner = _normalize_winner(d.get("winner_annotator", "A1"))
        confidence = _normalize_confidence(d.get("confidence", 0.50))
        confidence_level = _normalize_confidence_level(d.get("confidence_level", ""), confidence)

        return {
            "judge": self.judge_name,
            "view_mode": self.view_mode,
            "winner_annotator": winner,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "reasoning": d.get("reasoning", ""),
            "key_evidence": d.get("key_evidence", ""),
        }


class JudgeTiebreakerAgent:
    def __init__(self, judge_id: int, llm=None):
        self.judge_id = judge_id
        self.judge_name = f"Judge_{judge_id}"
        self.llm = llm if llm is not None else get_llm()
        self.prompt_template = create_judge_tiebreak_prompt_template()

    def decide(
        self,
        debate_result: Dict[str, Any],
        judge_1_result: Dict[str, Any],
        judge_2_result: Dict[str, Any],
        retrieved_context: str = "",
    ) -> Dict[str, Any]:
        review_text = debate_result.get("review_text", "")
        if not retrieved_context:
            retrieved_context = "(No specific guideline context retrieved.)"

        primary_sections = [
            {
                "title": "Primary Judge #1 Decision",
                "source_judge": judge_1_result.get("judge", "Judge_1"),
                "winner": judge_1_result.get("winner_annotator", "A1"),
                "confidence": judge_1_result.get("confidence", 0.50),
                "reasoning": judge_1_result.get("reasoning", ""),
                "key_evidence": judge_1_result.get("key_evidence", ""),
            },
            {
                "title": "Primary Judge #2 Decision",
                "source_judge": judge_2_result.get("judge", "Judge_2"),
                "winner": judge_2_result.get("winner_annotator", "A1"),
                "confidence": judge_2_result.get("confidence", 0.50),
                "reasoning": judge_2_result.get("reasoning", ""),
                "key_evidence": judge_2_result.get("key_evidence", ""),
            },
        ]
        random.shuffle(primary_sections)

        prompt = self.prompt_template.format(
            judge_name=self.judge_name,
            review_text=review_text,
            retrieved_context=retrieved_context,
            primary_section_title_1=primary_sections[0]["title"],
            primary_1_winner=primary_sections[0]["winner"],
            primary_1_confidence=primary_sections[0]["confidence"],
            primary_1_reasoning=primary_sections[0]["reasoning"],
            primary_1_key_evidence=primary_sections[0]["key_evidence"],
            primary_section_title_2=primary_sections[1]["title"],
            primary_2_winner=primary_sections[1]["winner"],
            primary_2_confidence=primary_sections[1]["confidence"],
            primary_2_reasoning=primary_sections[1]["reasoning"],
            primary_2_key_evidence=primary_sections[1]["key_evidence"],
        )

        messages = [HumanMessage(content=prompt)]
        raw_response = invoke_with_timeout_and_retry(
            self.llm,
            messages,
            agent_name=self.judge_name
        )
        parsed_resp = clean_json_output(raw_response.content)
        d = parsed_resp.get("decision", {})

        return {
            "judge": self.judge_name,
            "view_mode": "TIEBREAKER",
            "winner_annotator": _normalize_winner(d.get("winner_annotator", "A1")),
            "reasoning": d.get("reasoning", ""),
            "key_evidence": d.get("key_evidence", ""),
            "prompt_primary_section_order": [sec.get("source_judge", "") for sec in primary_sections],
            "prompt_primary_section_titles": [sec.get("title", "") for sec in primary_sections],
            "prompt_primary_winner_order": [sec.get("winner", "") for sec in primary_sections],
        }

def run_judge_panel(debate_result: Dict[str, Any], llm=None, retrieved_context: str = "") -> Dict[str, Any]:
    shared_llm = llm if llm is not None else get_llm()
    initial_positions = debate_result.get("initial_positions", {})

    primary_judges = [
        JudgeAgent(judge_id=1, llm=shared_llm, view_mode="A1_FIRST"),
        JudgeAgent(judge_id=2, llm=shared_llm, view_mode="A2_FIRST"),
    ]

    primary_results = []
    for judge in primary_judges:
        print(f"  -> {judge.judge_name} đang phân tích và phán quyết...")
        result = judge.decide(debate_result, retrieved_context=retrieved_context)
        primary_results.append(result)
        print(
            f"     Winner: {result['winner_annotator']} "
            f"| Confidence: {result['confidence']:.2f} "
            f"| Level: {result.get('confidence_level', 'MEDIUM')} "
            f"| View: {result['view_mode']}"
        )

    vote_count: Dict[str, int] = {}
    for jr in primary_results:
        ann = jr["winner_annotator"]
        if ann not in ["A1", "A2"]: ann = "A1"
        vote_count[ann] = vote_count.get(ann, 0) + 1

    j1 = primary_results[0]
    j2 = primary_results[1]

    final_decision_reason = ""
    selected_confidence = None
    all_judge_results = list(primary_results)
    tie_break_meta = {
        "invoked": False,
        "trigger": "not_needed",
        "prompt_primary_section_order": [],
        "prompt_primary_section_titles": [],
        "prompt_primary_winner_order": [],
        "winner_annotator": None,
    }

    if j1["winner_annotator"] == j2["winner_annotator"]:
        overall_winner = j1["winner_annotator"]
        winning_votes = 2
        verdict = "two_judges_consensus"
        selected_confidence = round((j1["confidence"] + j2["confidence"]) / 2, 2)
        final_decision_reason = "Both primary judges selected the same annotator."
    else:
        c1 = j1["confidence"]
        c2 = j2["confidence"]

        if c1 > c2:
            overall_winner = j1["winner_annotator"]
            winning_votes = 1
            verdict = "confidence_tiebreak"
            selected_confidence = c1
            final_decision_reason = "Primary judges disagreed; selected higher-confidence vote from Judge_1."
        elif c2 > c1:
            overall_winner = j2["winner_annotator"]
            winning_votes = 1
            verdict = "confidence_tiebreak"
            selected_confidence = c2
            final_decision_reason = "Primary judges disagreed; selected higher-confidence vote from Judge_2."
        else:
            print("  -> Judge_3 đang tie-break (2 judges disagree with equal confidence)...")
            tiebreak_judge = JudgeTiebreakerAgent(judge_id=3, llm=shared_llm)
            j3 = tiebreak_judge.decide(
                debate_result=debate_result,
                judge_1_result=j1,
                judge_2_result=j2,
                retrieved_context=retrieved_context,
            )
            all_judge_results.append(j3)
            overall_winner = j3["winner_annotator"]
            winning_votes = 1
            verdict = "judge3_tiebreak_equal_confidence"
            selected_confidence = c1
            final_decision_reason = "Primary judges disagreed with equal confidence; Judge_3 decided final winner."
            tie_break_meta = {
                "invoked": True,
                "trigger": "primary_disagreement_equal_confidence",
                "prompt_primary_section_order": j3.get("prompt_primary_section_order", []),
                "prompt_primary_section_titles": j3.get("prompt_primary_section_titles", []),
                "prompt_primary_winner_order": j3.get("prompt_primary_winner_order", []),
                "winner_annotator": j3.get("winner_annotator", overall_winner),
            }

    winner_labels = initial_positions.get(overall_winner, {}).get("labels", [])

    final_decision = {
        "winner_annotator": overall_winner,
        "winner_labels": winner_labels,
        "winning_votes": winning_votes,
        "verdict": verdict,
        "selected_confidence": selected_confidence,
        "decision_reason": final_decision_reason,
        "tie_break_meta": tie_break_meta,
    }

    return {
        "sample_id": debate_result.get("sample_id", ""),
        "review_text": debate_result.get("review_text", ""),
        "initial_positions": initial_positions,
        "judge_decisions": all_judge_results,
        "vote_summary": vote_count,
        "final_decision": final_decision,
    }