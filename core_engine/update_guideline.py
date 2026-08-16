import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from agents.guideline_agent import (
    append_to_guideline_file,
    propose_consolidated_guideline_update,
)


def interactive_update_guideline_batch(
    root_cause_records: list,
    current_guideline: str,
    target_domain: str,
    active_guideline_path: str,
):
    """Run one Guideline Agent call and one human review for an outer cycle."""
    proposal_json = propose_consolidated_guideline_update(
        root_cause_records=root_cause_records,
        current_guideline=current_guideline,
        target_domain=target_domain,
    )
    if not proposal_json or proposal_json.get("error"):
        print("[Guideline Agent] No valid consolidated report was produced.")
        return False

    cause_dir = os.path.join(ROOT_DIR, "system_data", "cause")
    os.makedirs(cause_dir, exist_ok=True)
    report_path = os.path.join(cause_dir, "guideline_cycle_suggestion.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(proposal_json, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(" CONSOLIDATED GUIDELINE REVIEW (HUMAN-IN-THE-LOOP) ")
    print("=" * 70)
    print(f"Root-cause records reviewed: {len(root_cause_records)}")
    print(f"Cycle summary: {proposal_json.get('cycle_summary', '')}")
    print(f"Full suggestion report: {report_path}")

    insights = proposal_json.get("key_insights", [])
    if isinstance(insights, list):
        for index, insight in enumerate(insights, start=1):
            if not isinstance(insight, dict):
                continue
            print(
                f"[{index}] {insight.get('recommended_action', 'NO_CHANGE')} | "
                f"{insight.get('pattern', '')}"
            )
            print(f"    Rationale: {insight.get('rationale', '')}")

    pending_file_path = os.path.join(ROOT_DIR, "system_data", "pending_rule.txt")
    candidate_text = proposal_json.get("candidate_guideline_text", "")
    with open(pending_file_path, "w", encoding="utf-8") as f:
        f.write(candidate_text if isinstance(candidate_text, str) else "")

    print(f"Editable candidate guideline text: {pending_file_path}")
    print("Review or edit the candidate text, then choose once for this cycle:")
    print("[1] Approve and apply the edited candidate text")
    print("[2] Reject all proposed edits for this cycle")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice != "1":
        print("[HITL] Consolidated guideline update rejected.")
        if os.path.exists(pending_file_path):
            os.remove(pending_file_path)
        return False

    with open(pending_file_path, "r", encoding="utf-8") as f:
        human_edited_text = f.read().strip()
    if not human_edited_text:
        print("[HITL] No candidate text remains after review; guideline unchanged.")
        os.remove(pending_file_path)
        return False

    proposal_json["candidate_guideline_text"] = human_edited_text
    proposal_json["action_type"] = "CONSOLIDATED UPDATE"
    proposal_json["location_in_guideline"] = "HUMAN-APPROVED CYCLE UPDATE"
    success = append_to_guideline_file(proposal_json, active_guideline_path)
    if os.path.exists(pending_file_path):
        os.remove(pending_file_path)
    return success


def process_all_causes(
    active_guideline_path: str,
    target_domain: str = "Restaurant",
):
    """Aggregate all root causes and trigger at most one human review."""
    cause_file_path = os.path.join(ROOT_DIR, "system_data", "cause", "cause_data.json")
    if not os.path.exists(cause_file_path):
        print(f"[CYCLE UPDATE] No root-cause file found: {cause_file_path}")
        return False

    with open(cause_file_path, "r", encoding="utf-8") as f:
        cause_list = json.load(f)
    if not isinstance(cause_list, list) or not cause_list:
        print("[CYCLE UPDATE] No root-cause records to consolidate.")
        return False

    try:
        with open(active_guideline_path, "r", encoding="utf-8") as f:
            full_guideline_content = f.read()
    except OSError as exc:
        print(f"[CYCLE UPDATE] Cannot read active guideline: {exc}")
        return False

    print(f"\n[CYCLE UPDATE] Consolidating {len(cause_list)} root-cause records.")
    result = interactive_update_guideline_batch(
        root_cause_records=cause_list,
        current_guideline=full_guideline_content,
        target_domain=target_domain,
        active_guideline_path=active_guideline_path,
    )
    print("[CYCLE UPDATE] Completed one consolidated human review.")
    return result
