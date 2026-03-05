import os
import json
from agents.adapt_agent import generate_adapted_guideline
from rag_system.build_knowledge_base import build_vector_database
from agents.annotator_agent import process_and_verify_review

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_vlsp_samples(file_path, num_samples=50):
    """Trích xuất mẫu từ dataset và trả về list để dễ dàng lặp qua từng câu."""
    samples = []
    if not os.path.exists(file_path):
        print(f"[CẢNH BÁO]: Không tìm thấy file dataset tại {file_path}")
        return []
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('#'):
                if i + 1 < len(lines):
                    content = lines[i+1].strip()
                    if content and not content.startswith('{'): 
                        samples.append(content)
            if len(samples) >= num_samples:
                break
    
    return samples

def run_workflow():
    data_dir = os.path.join(ROOT_DIR, "data")               
    system_data_dir = os.path.join(ROOT_DIR, "system_data") 
    
    os.makedirs(system_data_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, "1-VLSP2018-SA-Hotel-train (7-3-2018).txt")
    source_guideline_path = os.path.join(data_dir, "guideline.txt")
    adapted_guideline_path = os.path.join(system_data_dir, "adapted_guideline.txt")
    db_directory = os.path.join(system_data_dir, "chroma_db")
    
    # Đường dẫn file lưu các ca xung đột
    conflict_log_path = os.path.join(system_data_dir, "conflict_samples.jsonl")
    
    # Xóa file xung đột cũ (nếu có) để bắt đầu phiên chạy mới sạch sẽ
    if os.path.exists(conflict_log_path):
        os.remove(conflict_log_path)
    
    if not os.path.exists(source_guideline_path):
        print(f"[LỖI] KHÔNG TÌM THẤY: {source_guideline_path}. Vui lòng tạo file và thử lại.")
        return

    print(f"--- Đang trích xuất 50 mẫu từ {os.path.basename(dataset_path)} ---")
    sample_reviews_list = get_vlsp_samples(dataset_path, 50)
    
    if not sample_reviews_list:
        print("[LỖI] Không thể lấy mẫu từ dataset. Vui lòng kiểm tra định dạng file.")
        return
        
   
    print("ADAPT AGENT (DOMAIN ADAPTATION - HOTEL)")
   
    samples_str = "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(sample_reviews_list)])
    
    generate_adapted_guideline(
        source_file_path=source_guideline_path,
        target_domain="Hotel",
        samples=samples_str,
        output_file_path=adapted_guideline_path
    )
    print("\n")
    print("RAG SYSTEM (BUILD KNOWLEDGE BASE)")
    build_vector_database(
        file_path=adapted_guideline_path,
        persist_directory=db_directory
    )
    
    print("\n")
    print("ANNOTATOR AGENT (DUAL-AGENT LABELING)")
    
    # Chạy thử 5 câu đầu tiên
    test_samples = sample_reviews_list[:10] 
    
    agreed_count = 0
    conflict_count = 0

    for i, review in enumerate(test_samples):
        print(f"\n--- Đang xử lý câu {i+1}/{len(test_samples)} ---")
        print(f"Nội dung: '{review}'")
        
        result = process_and_verify_review(review_text=review, db_path=db_directory)
        
        if result["status"] == "AGREED":
            agreed_count += 1
            
        else:
            conflict_count += 1
            # Bổ sung: Lưu ca xung đột ra file JSONL
            with open(conflict_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result["data"], ensure_ascii=False) + "\n")
            print(f"[LƯU TRỮ] Đã ghi nhận ca xung đột vào {os.path.basename(conflict_log_path)}")
    print(f"Tổng số review đã xử lý: {len(test_samples)}")
    print(f"Số review ĐỒNG THUẬN (Đã lưu vào verified_samples.jsonl): {agreed_count}")
    print(f"Số review XUNG ĐỘT (Đã lưu vào conflict_samples.jsonl): {conflict_count}")

if __name__ == "__main__":
    run_workflow()