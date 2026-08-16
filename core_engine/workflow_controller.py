import os
import sys
import json
from datetime import datetime, timezone
from langchain_openai import AzureChatOpenAI
from langchain_chroma import Chroma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config.embedding_config import get_embeddings

from memory_and_history.history_manager import HistoryManager
from agents.debate_agent import create_debate_agents
from agents.summary_agent import create_summary_agent
from agents.judge_agent import run_judge_panel
from agents.root_cause_agent import analyze_root_cause

def get_knowledge_context(text: str) -> str:
    """Tự động lấy Guideline liên quan từ RAG"""
    # Sử dụng shared embedding instance
    embeddings = get_embeddings()
    db_path = os.path.join(ROOT_DIR, "system_data", "chroma_db")

    if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
        try:
            vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
            docs = vector_db.similarity_search(text, k=4)
            if docs:
                return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi truy xuất Knowledge DB: {e}")
    return "(No specific guideline context retrieved.)"


def _append_judge_audit_log(system_data_dir: str, judge_result: dict) -> str:
    """Append one judge audit record per sample to a JSONL file for traceability."""
    judge_decisions = judge_result.get("judge_decisions", [])
    final_decision = judge_result.get("final_decision", {})
    tie_break_meta = final_decision.get("tie_break_meta", {})

    primary_judges = [
        j for j in judge_decisions
        if j.get("judge") in ["Judge_1", "Judge_2"]
    ]
    primary_judges = sorted(primary_judges, key=lambda item: item.get("judge", ""))

    judge_3_result = next(
        (j for j in judge_decisions if j.get("judge") == "Judge_3"),
        None,
    )

    tie_break_info = {
        "invoked": bool(tie_break_meta.get("invoked", False) or judge_3_result is not None),
        "trigger": tie_break_meta.get("trigger", "not_needed"),
        "prompt_primary_section_order": tie_break_meta.get("prompt_primary_section_order", []),
        "prompt_primary_section_titles": tie_break_meta.get("prompt_primary_section_titles", []),
        "prompt_primary_winner_order": tie_break_meta.get("prompt_primary_winner_order", []),
        "winner_annotator": tie_break_meta.get("winner_annotator"),
        "judge_3_result": None,
    }

    if judge_3_result is not None:
        tie_break_info["judge_3_result"] = {
            "winner_annotator": judge_3_result.get("winner_annotator", "A1"),
            "reasoning": judge_3_result.get("reasoning", ""),
            "key_evidence": judge_3_result.get("key_evidence", ""),
        }
        if not tie_break_info["prompt_primary_section_order"]:
            tie_break_info["prompt_primary_section_order"] = judge_3_result.get("prompt_primary_section_order", [])
        if not tie_break_info["prompt_primary_section_titles"]:
            tie_break_info["prompt_primary_section_titles"] = judge_3_result.get("prompt_primary_section_titles", [])
        if not tie_break_info["prompt_primary_winner_order"]:
            tie_break_info["prompt_primary_winner_order"] = judge_3_result.get("prompt_primary_winner_order", [])

    log_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": judge_result.get("sample_id", ""),
        "review_text": judge_result.get("review_text", ""),
        "vote_summary": judge_result.get("vote_summary", {}),
        "primary_judges": [
            {
                "judge": j.get("judge", ""),
                "view_mode": j.get("view_mode", ""),
                "winner_annotator": j.get("winner_annotator", "A1"),
                "confidence": j.get("confidence"),
                "confidence_level": j.get("confidence_level"),
                "reasoning": j.get("reasoning", ""),
                "key_evidence": j.get("key_evidence", ""),
            }
            for j in primary_judges
        ],
        "tie_break": tie_break_info,
        "final_decision": {
            "winner_annotator": final_decision.get("winner_annotator", "A1"),
            "winning_votes": final_decision.get("winning_votes"),
            "verdict": final_decision.get("verdict", ""),
            "selected_confidence": final_decision.get("selected_confidence"),
            "decision_reason": final_decision.get("decision_reason", ""),
        },
    }

    os.makedirs(system_data_dir, exist_ok=True)
    log_file_path = os.path.join(system_data_dir, "judge_audit_log.jsonl")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return log_file_path

def run_full_conflict_workflow(
    conflict_data: dict,
    retriever=None,
    max_rounds: int = 4,
    enable_root_cause: bool = True,
):
    sample_id = conflict_data.get("sample_id", "Unknown")
    review_text = conflict_data.get("review", conflict_data.get("text", ""))

    A1_initial = conflict_data.get("A1", conflict_data.get("A1_initial", {}))
    A2_initial = conflict_data.get("A2", conflict_data.get("A2_initial", {}))

    print(f"\n[TIẾN TRÌNH] Bắt đầu luồng tranh biện và xử lý lỗi cho ca {sample_id}...")

    # BƯỚC MỚI: Truy xuất RAG 1 lần duy nhất cho toàn bộ luồng với error handling
    try:
        retrieved_context = get_knowledge_context(review_text)
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể lấy guideline context: {e}")
        retrieved_context = "(Không lấy được context từ guideline)"

    api_key_azure = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("BASE_URL") or os.getenv("AZURE_ENDPOINT")
    api_version = os.getenv("API_VERSION")
    deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")

    if not api_key_azure or not azure_endpoint:
        raise ValueError("[LỖI] Thiếu cấu hình Azure OpenAI trong .env")

    shared_llm = AzureChatOpenAI(
        api_key=api_key_azure,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=deployment_name,
        temperature=0.3
    )

    state = HistoryManager.initialize_histories(
        sample_id=sample_id,
        text=review_text,
        A1_initial=A1_initial,
        A2_initial=A2_initial,
        max_rounds=max_rounds
    )

    # Lưu RAG context vào state để Debate Agent dễ dàng lấy ra
    state["retrieved_context"] = retrieved_context

    debate_agents = create_debate_agents(llm=shared_llm)
    summary_agent = create_summary_agent(llm=shared_llm)

    state["current_case"] = "case_1"
    for _ in range(max_rounds):
        for agent_name in ["A1", "A2"]:
            state["current_turn"] = agent_name
            state = debate_agents[agent_name].generate_response(state)
            state = summary_agent.record_response(state)

    state["current_case"] = "case_2"
    for _ in range(max_rounds):
        for agent_name in ["A2", "A1"]:
            state["current_turn"] = agent_name
            state = debate_agents[agent_name].generate_response(state)
            state = summary_agent.record_response(state)

    state["current_case"] = "completed"
    state["current_turn"] = "moderator"
    final_history = HistoryManager.merge_results(state)

    system_data_dir = os.path.join(ROOT_DIR, "system_data")
    history_file_path = os.path.join(system_data_dir, "history_data.json")

    existing_histories = []
    if os.path.exists(history_file_path):
        try:
            with open(history_file_path, "r", encoding="utf-8") as f:
                existing_histories = json.load(f)
        except json.JSONDecodeError:
            existing_histories = []

    existing_histories.append(final_history)

    with open(history_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_histories, f, ensure_ascii=False, indent=4)

    print(f"\n[NHÁNH 1 - JUDGE] Đang phân xử để tìm người thắng...")

    # BƯỚC MỚI: Truyền luật xuống cho Judge Agent
    judge_result = run_judge_panel(final_history, llm=shared_llm, retrieved_context=retrieved_context)
    judge_audit_log_path = _append_judge_audit_log(system_data_dir, judge_result)
    print(f"[JUDGE LOG] Đã lưu audit vào: {judge_audit_log_path}")

    result_dir = os.path.join(system_data_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    winner = judge_result["final_decision"]["winner_annotator"]
    winner_data = {
        "review_id": sample_id,
        "review": review_text,
        "winner": winner,
        "labels": judge_result["final_decision"]["winner_labels"],
        "verdict": judge_result["final_decision"]["verdict"],
        "selected_confidence": judge_result["final_decision"].get("selected_confidence"),
        "decision_reason": judge_result["final_decision"].get("decision_reason", ""),
        "tie_break_invoked": judge_result["final_decision"].get("tie_break_meta", {}).get("invoked", False),
        "judge_audit_log_file": os.path.basename(judge_audit_log_path),
    }

    winner_file_path = os.path.join(result_dir, f"{sample_id}_WINNER_{winner}.json")
    with open(winner_file_path, "w", encoding="utf-8") as f:
        json.dump(winner_data, f, ensure_ascii=False, indent=4)

    if enable_root_cause:
        print("\n[ROOT CAUSE] Running root-cause analysis...")
        root_cause_result = analyze_root_cause(final_history)
    else:
        print("[ROOT CAUSE] Skipped by annotation-only mode.")
        root_cause_result = None

    return {
        "final_history": final_history,
        "judge_result": judge_result,
        "root_cause_result": root_cause_result
    }
