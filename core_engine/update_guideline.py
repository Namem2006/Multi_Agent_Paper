import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from agents.guideline_agent import propose_guideline_update, append_to_guideline_file
from rag_system.build_knowledge_base import build_vector_database

def interactive_update_guideline(root_cause_data: dict, current_guideline_chunk: str, target_domain: str, guideline_name: str):
    """
    Ham tuong tac: Lay de xuat luat moi va hoi nguoi dung co muon luu hay khong.
    """
    guideline_filepath = os.path.join(ROOT_DIR, "data", guideline_name)
    db_directory = os.path.join(ROOT_DIR, "system_data", "chroma_db")
    
    review_id = root_cause_data.get("review_id", "Unknown")
    print(f"\n[HE THONG] Dang phan tich va de xuat luat moi cho ca xung dot {review_id}...")

    proposal_json = propose_guideline_update(root_cause_data, current_guideline_chunk, target_domain)

    if not proposal_json:
        print("-> Khong co de xuat cap nhat luat nao duoc tao ra.")
        return False

    option_1 = proposal_json.get("option_1_direct_content", "")
    option_2 = proposal_json.get("option_2_proposal_note", "")
    target_section = proposal_json.get("target_section", "NEW RULES")

    print("\n" + "="*70)
    print(" QUYET DINH CAP NHAT GUIDELINE ")
    print("="*70)
    print(f"Vi tri de xuat chen vao: {target_section}\n")
    print("[OPTION 1 - DIRECT FIX] Noi dung san sang de chen:")
    print(f"   {option_1}\n")
    print("[OPTION 2 - CONSULTATION] Giai thich cua AI (Danh cho ban tham khao):")
    print(f"   {option_2}\n")
    print("="*70)
    
    print("Lua chon hanh dong:")
    print("[1]. CHAP NHAN: Tu dong chen Option 1 vao file va nap lai vao Vector DB (Chroma).")
    print("[2]. BO QUA / TU CHOI: Khong ghi file cho ca nay.")
    
    choice = input("Nhap lua chon cua ban (1 hoac 2): ").strip()

    if choice == '1':
        print("\n[Tien trinh] Dang ghi luat vao file...")
        success = append_to_guideline_file(proposal_json, guideline_filepath)
        if success:
            print("[Tien trinh] Dang bam (embed) va cap nhat lai Vector Database...")
            build_vector_database(guideline_filepath, db_directory)
            print("[HOAN TAT] Luat moi da duoc tham thau vao he thong RAG!")
            return True
        else:
            print("[LOI] Cap nhat that bai do loi ghi file.")
            return False
            
    elif choice == '2':
        print("\n[TAM DUNG] He thong khong tu dong ghi ca nay.")
        return False
        
    else:
        print("\nLua chon khong hop le. Da huy thao tac cap nhat cho ca nay.")
        return False

def process_all_causes():
    cause_file_path = os.path.join(ROOT_DIR, "system_data", "cause", "cause_data.json")
    
    if not os.path.exists(cause_file_path):
        print(f"[LOI] Khong tim thay file {cause_file_path}. Hay chay Root Cause Agent truoc.")
        return

    with open(cause_file_path, "r", encoding="utf-8") as f:
        cause_list = json.load(f)

    if not cause_list:
        print("[THONG BAO] Khong co du lieu xung dot nao de xu ly.")
        return

    print(f"\n[BAT DAU] Tim thay {len(cause_list)} ca xung dot can xem xet.")
    
    mock_current_guideline = "- DRINKS#QUALITY: Dung cho cac mo ta ve chat luong do uong."
    
    for cause_data in cause_list:
        if cause_data.get("need_update", False):
            interactive_update_guideline(
                root_cause_data=cause_data, 
                current_guideline_chunk=mock_current_guideline, 
                target_domain="Restaurant",
                guideline_name="guideline.txt"
            )
        else:
            review_id = cause_data.get("review_id", "Unknown")
            print(f"\n[BO QUA] Ca {review_id} khong can cap nhat luat theo danh gia cua Root Cause Agent.")
            
    print("\n[HOAN THANH] Da duyet qua toan bo danh sach xung dot.")

if __name__ == "__main__":
    process_all_causes()