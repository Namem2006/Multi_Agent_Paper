import os
import sys
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config.embedding_config import get_embeddings

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def build_vector_database(file_path: str, persist_directory: str):
    embeddings = get_embeddings()

    # Dọn dẹp Database cũ
    if os.path.exists(persist_directory):
        print(f"Đang dọn dẹp Database cũ tại {persist_directory}...")
        try:
            old_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            old_db.delete_collection()
            print("Đã xóa collection cũ.")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể xóa collection cũ: {e}")
            import shutil
            try:
                shutil.rmtree(persist_directory)
                print("Đã xóa thư mục database cũ.")
            except Exception as e2:
                print(f"[LỖI] Không thể xóa thư mục: {e2}")

    print(f"[RAG System] Đang đọc dữ liệu từ: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # 1. BƯỚC 1: Cắt theo cấu trúc Markdown để gắn thẻ Metadata (Biết được đoạn text nằm ở mục nào)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(document_text)

    # 2. BƯỚC 2: SEMANTIC CHUNKING (Gom cụm câu theo ngữ nghĩa)
    print("[RAG System] Đang phân tích ngữ nghĩa để chia chunk (Semantic Chunking)...")
    semantic_chunker = SemanticChunker(
        embeddings,
        # Thuật toán tách chunk dựa trên phần trăm khác biệt ngữ nghĩa.
        # Cứ khi nào độ lệch nghĩa giữa 2 câu liên tiếp vượt quá 85% so với tổng thể, nó sẽ cắt.
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )

    # Áp dụng Semantic Chunking lên các đoạn đã có metadata từ Markdown
    final_chunks = semantic_chunker.split_documents(md_header_splits)

    # 3. Tạo và lưu Vector DB
    Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"[RAG System] Đã nạp {len(final_chunks)} chunks NGỮ NGHĨA vào Vector DB tại: {persist_directory}")

if __name__ == "__main__":
    # Test chạy thử độc lập
    file_path = os.path.join(ROOT_DIR, "guideline.txt")
    db_path = os.path.join(ROOT_DIR, "system_data", "chroma_db")
    build_vector_database(file_path, db_path)