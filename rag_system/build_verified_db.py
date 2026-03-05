import os
import sys
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def save_verified_sample(verified_data: dict):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[LỖI] Không tìm thấy GOOGLE_API_KEY trong file .env")
        return

    # 1. Định nghĩa đường dẫn lưu trữ
    system_data_dir = os.path.join(ROOT_DIR, "system_data")
    os.makedirs(system_data_dir, exist_ok=True)
    
    jsonl_path = os.path.join(system_data_dir, "verified_samples.jsonl")
    persist_dir = os.path.join(system_data_dir, "chroma_db_verified")

    # 2. Ghi nối thêm vào file JSONL
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(verified_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LỖI] Không thể ghi vào file JSONL: {str(e)}")
        return

    # 3. Tạo Document để nạp vào Vector DB
    page_content = f"""
    --- VERIFIED CASE ---
    Review: "{verified_data.get('review', '')}"
    Correct Label: {json.dumps(verified_data.get('labels', []), ensure_ascii=False)}
    """
    
    meta = {
        "source": "auto_consensus",
        "status": "verified"
    }
    
    doc = Document(page_content=page_content, metadata=meta)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Nạp vào ChromaDB dành riêng cho dữ liệu sạch (Verified DB)
    try:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vectorstore.add_documents([doc])
        print("[RAG SYSTEM] Đã cập nhật mẫu đồng thuận vào Verified DB.")
    except Exception as e:
        print(f"[LỖI] Không thể nạp vào Verified ChromaDB: {str(e)}")

def build_verified_db_from_scratch():
    """Hàm tiện ích: Đọc lại toàn bộ file JSONL và xây lại DB từ đầu (nếu cần)"""
    jsonl_path = os.path.join(ROOT_DIR, "system_data", "verified_samples.jsonl")
    persist_dir = os.path.join(ROOT_DIR, "system_data", "chroma_db_verified")
    
    if not os.path.exists(jsonl_path):
        print(f"[CẢNH BÁO] Không tìm thấy file dữ liệu chuẩn: {jsonl_path}")
        return

    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                page_content = f"""
                --- VERIFIED CASE ---
                Review: "{item.get('review', '')}"
                Correct Label: {json.dumps(item.get('labels', []), ensure_ascii=False)}
                """
                doc = Document(page_content=page_content, metadata={"source": "batch_import"})
                documents.append(doc)

    if documents:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        print(f"Đang tạo lại database chuẩn tại {persist_dir} với {len(documents)} mẫu...")
        Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
        print("Đã xây dựng xong kho dữ liệu mẫu chuẩn.")

if __name__ == "__main__":
    build_verified_db_from_scratch()