import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def compare_annotations(gemini_labels, gpt_labels):
    """So sánh 2 kết quả. Bỏ qua thứ tự, chỉ quan tâm Entity, Attribute, Sentiment."""
    def extract_core_elements(labels):
        elements = set()
        for item in labels:
            if isinstance(item, dict):
                # Lấy giá trị ra, nếu là None thì biến thành chuỗi rỗng
                ent = item.get("entity")
                attr = item.get("attribute")
                sent = item.get("sentiment")

                # Xử lý an toàn: Ép kiểu sang string, xóa khoảng trắng, in hoa
                ent_clean = str(ent).strip().upper() if ent is not None else ""
                attr_clean = str(attr).strip().upper() if attr is not None else ""
                sent_clean = str(sent).strip().upper() if sent is not None else ""
                
                elements.add((ent_clean, attr_clean, sent_clean))
        return elements

    gemini_set = extract_core_elements(gemini_labels)
    gpt_set = extract_core_elements(gpt_labels)

    return gemini_set == gpt_set

def filter_and_route_conflict(review_text, gemini_result, gpt_result):
    """Lọc kết quả: Nếu giống nhau thì lưu DB, nếu khác thì đẩy ra danh sách xung đột"""
    is_match = compare_annotations(gemini_result, gpt_result)

    if is_match:
        print("[THÀNH CÔNG] HAI AGENT ĐỒNG THUẬN! Dữ liệu đạt chuẩn.")
        verified_data = {
            "review": review_text,
            "labels": gemini_result 
        }
        
        try:
            from rag_system.build_verified_db import save_verified_sample
            save_verified_sample(verified_data)
            print("[LƯU TRỮ] Đã lưu vào Verified Database.")
        except ImportError:
            print("[CẢNH BÁO] Chưa có hàm save_verified_sample trong build_verified_db.py.")
            
        return {"status": "AGREED", "data": verified_data}
        
    else:
        print("[XUNG ĐỘT] PHÁT HIỆN XUNG ĐỘT! Đưa vào danh sách chờ tranh biện (Debate).")
        conflict_data = {
            "review": review_text,
            "gemini_labels": gemini_result,
            "gpt_labels": gpt_result
        }
        return {"status": "CONFLICT", "data": conflict_data}