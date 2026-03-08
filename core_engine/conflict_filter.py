import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def compare_annotations(gemini_labels, gpt_labels):
    def extract_core_elements(labels):
        elements = set()
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

    gemini_set = extract_core_elements(gemini_labels)
    gpt_set = extract_core_elements(gpt_labels)

    return gemini_set == gpt_set

def filter_and_route_conflict(review_id, review_text, gemini_result, gpt_result):
    is_match = compare_annotations(gemini_result, gpt_result)
    
    system_data_dir = os.path.join(ROOT_DIR, "system_data")
    result_dir = os.path.join(system_data_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    if is_match:
        print("[THÀNH CÔNG] HAI AGENT ĐỒNG THUẬN! Dữ liệu đạt chuẩn.")
        verified_data = {
            "review_id": review_id,
            "review": review_text,
            "labels": gemini_result 
        }
        
        #Lưu vào Verified DB (Kho dữ liệu chuẩn)
        try:
            from rag_system.build_verified_db import save_verified_sample
            save_verified_sample(verified_data)
            print("[LƯU TRỮ] Đã lưu vào Verified Database.")
        except ImportError:
            print("[CẢNH BÁO] Chưa có hàm save_verified_sample trong build_verified_db.py.")
        #Lưu vào result để sau này tính độ hiệu quả của mô hình    
        result_file_path = os.path.join(result_dir, f"{review_id}_AGREED.json")
        try:
            with open(result_file_path, "w", encoding="utf-8") as f:
                json.dump(verified_data, f, ensure_ascii=False, indent=4)
            print(f"[LƯU TRỮ] Đã lưu kết quả chi tiết vào: result/{review_id}_AGREED.json")
        except Exception as e:
            print(f"[LỖI] Không thể lưu file result: {e}")

        return {"status": "AGREED", "data": verified_data}
        
    else:
        print("[XUNG ĐỘT] PHÁT HIỆN XUNG ĐỘT! Đưa vào danh sách chờ tranh biện (Debate).")
        conflict_data = {
            "review_id": review_id,
            "review": review_text,
            "gemini_labels": gemini_result,
            "gpt_labels": gpt_result
        }
        
        # 1. Lưu vào file conflict_samples.jsonl (Để chạy Debate hàng loạt sau này)
        conflict_log_path = os.path.join(system_data_dir, "conflict_samples.jsonl")
        try:
            with open(conflict_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(conflict_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[LỖI] Không thể lưu file conflict_samples.jsonl: {e}")

        return {"status": "CONFLICT", "data": conflict_data}