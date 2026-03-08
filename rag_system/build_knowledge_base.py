import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def build_vector_database(file_path: str, persist_directory: str):
    if os.path.exists(persist_directory):
        print(f"Đang dọn dẹp Database cũ tại {persist_directory}...")
        shutil.rmtree(persist_directory)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("LỖI: Không tìm thấy GOOGLE_API_KEY")
        return

    print(f"[RAG System] Đang đọc và băm dữ liệu từ: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Tạo và lưu Vector DB
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"[RAG System] Đã nạp {len(chunks)} chunks vào Vector DB tại: {persist_directory}")