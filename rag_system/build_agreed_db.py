import os
import sys
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config.embedding_config import get_embeddings

load_dotenv(os.path.join(ROOT_DIR, ".env"))

AGREED_SAMPLES_FILENAME = "agreed_samples.jsonl"
AGREED_DB_DIRNAME = "chroma_db_agreed"


def save_agreed_sample(agreed_data: dict):
    # 1. Định nghĩa đường dẫn lưu trữ
    system_data_dir = os.path.join(ROOT_DIR, "system_data")
    os.makedirs(system_data_dir, exist_ok=True)

    jsonl_path = os.path.join(system_data_dir, AGREED_SAMPLES_FILENAME)
    persist_dir = os.path.join(system_data_dir, AGREED_DB_DIRNAME)

    # 2. Ghi nối thêm vào file JSONL
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(agreed_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LỖI] Không thể ghi vào file JSONL: {str(e)}")
        return

    # 3. Tạo Document để nạp vào Vector DB
    page_content = f"""
    --- AGREED CASE ---
    Review: "{agreed_data.get('review', '')}"
    Dual-agent agreed label: {json.dumps(agreed_data.get('labels', []), ensure_ascii=False)}
    """

    meta = {
        "source": "auto_consensus",
        "status": "dual_agent_agreed"
    }

    doc = Document(page_content=page_content, metadata=meta)

    # Sử dụng shared embedding instance
    embeddings = get_embeddings()

    # 4. Nạp vào ChromaDB dành riêng cho agreed-case memory.
    try:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vectorstore.add_documents([doc])
        print("[RAG SYSTEM] Đã cập nhật mẫu đồng thuận vào agreed-case DB.")
    except Exception as e:
        error_msg = str(e)
        print(f"[LỖI] Không thể nạp vào agreed-case ChromaDB: {error_msg}")

        # Nếu lỗi do dimension mismatch, rebuild database từ đầu
        if "dimension" in error_msg.lower():
            print("[CẢNH BÁO] Phát hiện lỗi dimension mismatch. Đang rebuild database...")
            build_agreed_db_from_scratch()

def build_agreed_db_from_scratch():
    """Hàm tiện ích: Đọc lại toàn bộ file JSONL và xây lại DB từ đầu (nếu cần)"""
    jsonl_path = os.path.join(ROOT_DIR, "system_data", AGREED_SAMPLES_FILENAME)
    persist_dir = os.path.join(ROOT_DIR, "system_data", AGREED_DB_DIRNAME)

    if not os.path.exists(jsonl_path):
        print(f"[CẢNH BÁO] Không tìm thấy file dữ liệu chuẩn: {jsonl_path}")
        return

    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                page_content = f"""
                --- AGREED CASE ---
                Review: "{item.get('review', '')}"
                Dual-agent agreed label: {json.dumps(item.get('labels', []), ensure_ascii=False)}
                """
                doc = Document(page_content=page_content, metadata={"source": "batch_import"})
                documents.append(doc)

    if documents:
        # Sử dụng shared embedding instance
        embeddings = get_embeddings()

        # XÓA DATABASE CŨ TRƯỚC KHI TẠO MỚI (tránh lỗi dimension mismatch)
        if os.path.exists(persist_dir):
            print(f"Đang xóa database cũ tại {persist_dir}...")
            try:
                old_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
                old_db.delete_collection()
                print("Đã xóa database cũ.")
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể xóa collection cũ: {e}")
                # Nếu không xóa được, thử xóa thư mục vật lý
                import shutil
                try:
                    shutil.rmtree(persist_dir)
                    print("Đã xóa thư mục database cũ.")
                except Exception as e2:
                    print(f"[LỖI] Không thể xóa thư mục: {e2}")

        print(f"Đang tạo lại database chuẩn tại {persist_dir} với {len(documents)} mẫu...")
        Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
        print("Đã xây dựng xong kho dữ liệu mẫu chuẩn.")

if __name__ == "__main__":
    build_agreed_db_from_scratch()
