import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def compare_annotations(a1_data, a2_data):
    def extract_core_elements(data):
        elements = set()
        labels = data.get("labels", []) if isinstance(data, dict) else []
        for item in labels:
            if isinstance(item, dict):
                ent = item.get("entity")
                attr = item.get("attribute")
                sent = item.get("sentiment")

                ent_clean = str(ent).strip().upper() if ent is not None else ""
                attr_clean = str(attr).strip().upper() if attr is not None else ""
                sent_clean = str(sent).strip().upper() if sent is not None else ""

                elements.add((ent_clean, attr_clean, sent_clean))
        return elements

    a1_set = extract_core_elements(a1_data)
    a2_set = extract_core_elements(a2_data)

    return a1_set == a2_set

def filter_and_route_conflict(review_id, review_text, a1_result, a2_result):
    is_match = compare_annotations(a1_result, a2_result)

    system_data_dir = os.path.join(ROOT_DIR, "system_data")
    result_dir = os.path.join(system_data_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    if is_match:
        print("[THANH CONG] HAI AGENT DONG THUAN! Du lieu dat chuan.")
        agreed_data = {
            "review_id": review_id,
            "review": review_text,
            "labels": a1_result.get("labels", []),
            # SỬA LỖI: Bổ sung lưu trữ opinion và evidence
            "opinion": a1_result.get("opinion", ""),
            "evidence": a1_result.get("evidence", "")
        }

        try:
            from rag_system.build_agreed_db import save_agreed_sample
            save_agreed_sample(agreed_data)
            print("[LUU TRU] Da luu vao agreed-case database.")
        except ImportError:
            print("[CANH BAO] Chua co ham save_agreed_sample trong build_agreed_db.py.")

        result_file_path = os.path.join(result_dir, f"{review_id}_AGREED.json")
        try:
            with open(result_file_path, "w", encoding="utf-8") as f:
                json.dump(agreed_data, f, ensure_ascii=False, indent=4)
            print(f"[LUU TRU] Da luu ket qua chi tiet vao: result/{review_id}_AGREED.json")
        except Exception as e:
            print(f"[LOI] Khong the luu file result: {e}")

        return {"status": "AGREED", "data": agreed_data}

    else:
        print("[XUNG DOT] PHAT HIEN XUNG DOT! Dua vao danh sach cho tranh bien (Debate).")
        conflict_data = {
            "review_id": review_id,
            "review": review_text,
            "a1_labels": a1_result.get("labels", []),
            "a2_labels": a2_result.get("labels", []),
            # Bắt giữ toàn bộ lập luận để truyền cho Debate
            "a1_opinion": a1_result.get("opinion", ""),
            "a1_evidence": a1_result.get("evidence", ""),
            "a2_opinion": a2_result.get("opinion", ""),
            "a2_evidence": a2_result.get("evidence", "")
        }

        conflict_log_path = os.path.join(system_data_dir, "conflict_samples.jsonl")
        try:
            with open(conflict_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(conflict_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[LOI] Khong the luu file conflict_samples.jsonl: {e}")

        return {"status": "CONFLICT", "data": conflict_data}
