import os
import json
import math
from agents.adapt_agent import generate_adapted_guideline
from rag_system.build_knowledge_base import build_vector_database
# IMPORT HÀM XỬ LÝ BATCH
from agents.annotator_agent import process_and_verify_batch
from rag_system.build_verified_db import build_verified_db_from_scratch
# IMPORT HÀM ĐỌC DATA MỚI TẠO
from core_engine.data_loader import extract_and_assign_ids

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_workflow():
    data_dir = os.path.join(ROOT_DIR, "data")               
    system_data_dir = os.path.join(ROOT_DIR, "system_data") 
    os.makedirs(system_data_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, "1-VLSP2018-SA-Restaurant-train (7-3-2018).txt")
    source_guideline_path = os.path.join(data_dir, "guideline.txt")
    adapted_guideline_path = os.path.join(system_data_dir, "adapted_guideline.txt")
    
    db_directory = os.path.join(system_data_dir, "chroma_db") 
    conflict_log_path = os.path.join(system_data_dir, "conflict_samples.jsonl")
    
    if os.path.exists(conflict_log_path):
        os.remove(conflict_log_path)
    
    print(" HỆ THỐNG GÁN NHÃN ĐA TÁC VỤ (MULTI-AGENT ACSA)")
    print("Chọn chế độ chạy:")
    print("[1]. Domain mới (Chạy Adapt Agent tạo luật)")
    print("[2]. Domain đã có guideline (Bỏ qua tạo luật)")
    
    choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()
    
    print(f"\n--- Đang trích xuất toàn bộ dữ liệu từ {os.path.basename(dataset_path)} ---")
    # SỬ DỤNG HÀM TỪ DATA_LOADER
    all_sample_reviews = extract_and_assign_ids(dataset_path)
    
    # CHỈ LẤY 30 CÂU ĐỂ CHẠY CHO MỖI LẦN RUN
    run_samples = all_sample_reviews[:30]
    total_samples = len(run_samples)
    
    if total_samples == 0:
        print("[LỖI] Không thể lấy mẫu từ dataset hoặc file trống.")
        return
    else:
        print(f"-> Đã lấy thành công {total_samples} câu review cho phiên chạy này.")

    active_guideline_path = source_guideline_path 
    if choice == '1':
        if not os.path.exists(source_guideline_path):
            print(f"[LỖI] KHÔNG TÌM THẤY: {source_guideline_path}. Vui lòng tạo file và thử lại.")
            return
            
        print("\nADAPT AGENT (DOMAIN ADAPTATION)")
        # Lấy tối đa 50 câu để làm mẫu tạo luật (dùng all_sample_reviews để có đa dạng)
        adapt_samples = all_sample_reviews[:50]
        samples_str = "\n".join([f"{idx+1}. {item['text']}" for idx, item in enumerate(adapt_samples)])
        
        generate_adapted_guideline(
            source_file_path=source_guideline_path,
            target_domain="Hotel", # Hoặc Restaurant tùy dữ liệu
            samples=samples_str,
            output_file_path=adapted_guideline_path
        )
        active_guideline_path = adapted_guideline_path
    else:
        print("\n[BƯỚC 1] BỎ QUA ADAPT AGENT (Sử dụng luật có sẵn).")
        if os.path.exists(adapted_guideline_path):
            active_guideline_path = adapted_guideline_path
            print(f"-> Đang sử dụng file luật: {os.path.basename(adapted_guideline_path)}")
        elif os.path.exists(source_guideline_path):
            active_guideline_path = source_guideline_path
            print(f"-> Đang sử dụng file luật gốc: {os.path.basename(source_guideline_path)}")
        else:
            print("[LỖI] Không tìm thấy bất kỳ file luật nào!")
            return

    print("\n[BƯỚC 2] RAG SYSTEM (BUILD KNOWLEDGE BASE)")
    build_vector_database(
        file_path=active_guideline_path,
        persist_directory=db_directory
    )
    chunk_size = 3  
    total_turns = math.ceil(total_samples / chunk_size) # Ví dụ 30 câu -> 10 turns
    
    global_agreed = 0
    global_conflict = 0
    
    print(f"\n[BƯỚC 3] ANNOTATOR AGENT BẮT ĐẦU CHẠY ({total_turns} Turns, mỗi turn {chunk_size} câu)")

    for turn in range(total_turns):
        start_idx = turn * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        current_batch = run_samples[start_idx:end_idx] # Mảng 3 object dict
        
        print(f"\n{'='*60}")
        print(f" 🔄 TURN {turn + 1}/{total_turns} | Đang xử lý câu {start_idx + 1} đến {end_idx} ")
        print(f"{'='*60}")
        
        # In tóm tắt các câu đang được xử lý trong Batch
        for item in current_batch:
            short_text = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
            print(f"--- ID: {item['id']} | {short_text}")
            
        # Gọi thẳng hàm Batch Process
        results = process_and_verify_batch(batch_data=current_batch, base_db_dir=system_data_dir)
        
        turn_agreed = 0
        turn_conflict = 0
        
        # Hàm trả về List các result tương ứng từng câu, ta lặp qua để đếm
        for res in results:
            if res["status"] == "AGREED":
                turn_agreed += 1
                global_agreed += 1
            else:
                turn_conflict += 1
                global_conflict += 1

        print(f"\n-> Kết thúc TURN {turn + 1}. Đồng thuận: {turn_agreed} | Xung đột: {turn_conflict}")
        
        # BƯỚC 4: CẬP NHẬT ÁN LỆ SAU MỖI TURN NẾU CÓ CÂU ĐỒNG THUẬN
        if turn_agreed > 0:
            print(f"\n[BƯỚC 4] NẠP {turn_agreed} ÁN LỆ MỚI VÀO VERIFIED DATABASE...")
            build_verified_db_from_scratch()
        else:
            print("\n[BƯỚC 4] Không có ca đồng thuận nào trong Turn này. Bỏ qua cập nhật Verified DB.")

    # ==========================================
    # TỔNG KẾT SAU KHI CHẠY HẾT PHIÊN (30 CÂU)
    # ==========================================
    print("\n" + "★"*50)
    print(" HOÀN THÀNH QUÁ TRÌNH XỬ LÝ ")
    print("★"*50)
    print(f"Tổng số review đã xử lý: {total_samples}")
    print(f"Tổng số ĐỒNG THUẬN (Sạch) : {global_agreed}")
    print(f"Tổng số XUNG ĐỘT (Debate): {global_conflict}")
    print(f"-> Kiểm tra thư mục 'system_data/result/' để xem kết quả chi tiết từng câu.")

if __name__ == "__main__":
    run_workflow()