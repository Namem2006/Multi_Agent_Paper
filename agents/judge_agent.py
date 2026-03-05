"""
Judge Agent for ACSA Debate System

3 independent judges each review the full debate history and pick ONE overall
winner annotator (A1 or A2) — the one with the more correct full label set.
Majority voting determines the final winner.

Workflow:
    debate_results.json
         |
    JudgeAgent(1).decide()  ->  { winner_annotator: "A1", reasoning: ..., key_evidence: ... }
    JudgeAgent(2).decide()  ->  { winner_annotator: "A1", ... }
    JudgeAgent(3).decide()  ->  { winner_annotator: "A2", ... }
         |
    run_judge_panel()  ->  majority vote  ->  final_decision
"""

import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage

from models.schemas import MultiLabelJudgeResponse
from config.llm_config import get_llm, GUIDELINE_CONTENT
from prompts.judge_prompts import create_judge_prompt_template


class JudgeAgent:
    """
    A single judge agent that independently reviews the debate and decides
    which annotator (A1 or A2) has the OVERALL more correct label set.
    """

    def __init__(self, judge_id: int, llm=None):
        self.judge_id = judge_id
        self.judge_name = f"Judge_{judge_id}"
        self.llm = llm if llm is not None else get_llm()
        self.prompt_template = create_judge_prompt_template()

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        lines = []
        for entry in history:
            round_num = entry.get("round", "?")
            for agent_name in ["A1", "A2"]:
                if agent_name in entry:
                    data = entry[agent_name]
                    opinion = data.get("opinion", "")
                    evidence = data.get("evidence", "")
                    if "labels" in data:
                        labels_str = ", ".join(
                            f"{l['entity']}#{l['attribute']}|{l['sentiment']}"
                            for l in data["labels"]
                        )
                    elif "label" in data:
                        lbl = data["label"]
                        labels_str = f"{lbl['entity']}#{lbl['attribute']}|{lbl['sentiment']}"
                    else:
                        labels_str = "(unknown)"
                    block = (
                        f"[Round {round_num} - {agent_name}]\n"
                        f"  Labels: {labels_str}\n"
                        f"  Opinion: {opinion}\n"
                        f"  Evidence: {evidence}"
                    )
                    lines.append(block)
        return "\n\n".join(lines) if lines else "(Không có lịch sử)"

    def _format_labels_text(self, labels: List[Dict[str, Any]]) -> str:
        lines = []
        for i, lbl in enumerate(labels, start=1):
            lines.append(
                f"  {i}. {lbl['entity']}#{lbl['attribute']} | {lbl['sentiment']}"
            )
        return "\n".join(lines) if lines else "  (Không có nhãn)"

    def decide(self, debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review the full debate result and return ONE overall winner decision.

        Args:
            debate_result: Full output from debate phase

        Returns:
            Dict with judge_name, winner_annotator, reasoning, key_evidence
        """
        review_text = debate_result.get("review_text", "")
        initial_positions = debate_result.get("initial_positions", {})

        a1_labels = initial_positions.get("A1", {}).get("labels", [])
        a2_labels = initial_positions.get("A2", {}).get("labels", [])

        a1_labels_text = self._format_labels_text(a1_labels)
        a2_labels_text = self._format_labels_text(a2_labels)

        case_1_history = self._format_history(
            debate_result.get("debate_summary", {}).get("case_1", {}).get("history", [])
        )
        case_2_history = self._format_history(
            debate_result.get("debate_summary", {}).get("case_2", {}).get("history", [])
        )

        prompt = self.prompt_template.format(
            judge_name=self.judge_name,
            review_text=review_text,
            a1_labels_text=a1_labels_text,
            a2_labels_text=a2_labels_text,
            guideline_content=GUIDELINE_CONTENT,
            case_1_history=case_1_history,
            case_2_history=case_2_history,
        )

        messages = [HumanMessage(content=prompt)]
        response: MultiLabelJudgeResponse = self.llm.with_structured_output(
            MultiLabelJudgeResponse
        ).invoke(messages)

        d = response.decision
        return {
            "judge": self.judge_name,
            "winner_annotator": d.winner_annotator,
            "reasoning": d.reasoning,
            "key_evidence": d.key_evidence,
        }


def run_judge_panel(debate_result: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """
    Run all 3 judges and perform majority voting to pick the overall winner.

    Args:
        debate_result: Full debate output dict (with initial_positions)
        llm: Optional shared LLM instance

    Returns:
        Dict with all 3 judge responses, vote summary, and final_decision
    """
    shared_llm = llm if llm is not None else get_llm()
    initial_positions = debate_result.get("initial_positions", {})

    judges = [JudgeAgent(judge_id=i, llm=shared_llm) for i in range(1, 4)]

    all_judge_results = []
    for judge in judges:
        print(f"  -> {judge.judge_name} đang phân tích và phán quyết...")
        result = judge.decide(debate_result)
        all_judge_results.append(result)
        print(f"     Winner: {result['winner_annotator']}")

    # Majority vote on winner_annotator
    vote_count: Dict[str, int] = {}
    for jr in all_judge_results:
        ann = jr["winner_annotator"]
        vote_count[ann] = vote_count.get(ann, 0) + 1

    overall_winner = max(vote_count, key=lambda k: vote_count[k])
    winning_votes = vote_count[overall_winner]
    verdict = "unanimous" if winning_votes == 3 else "majority" if winning_votes >= 2 else "no_majority"

    winner_labels = initial_positions.get(overall_winner, {}).get("labels", [])

    final_decision = {
        "winner_annotator": overall_winner,
        "winner_labels": winner_labels,
        "winning_votes": winning_votes,
        "verdict": verdict,
    }

    return {
        "sample_id": debate_result.get("sample_id", ""),
        "review_text": debate_result.get("review_text", ""),
        "initial_positions": initial_positions,
        "judge_decisions": all_judge_results,
        "vote_summary": vote_count,
        "final_decision": final_decision,
    }
