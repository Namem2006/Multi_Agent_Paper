import json
import os
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


GOOGLE_API_KEY = "AIzaSyBrwzeGRALqrf2Hdl0s7cXnwr6QqqpOk-Q"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def build_verified_db():
    json_path = r"D:\Project\multi agent\work\verified_gold_samples.json"
    persist_dir = r"D:\Project\multi agent\work\RAG\chroma_db_verified_examples"

    if not os.path.exists(json_path):
        print(f"❌ Không tìm thấy file dữ liệu chuẩn: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    

    documents = []
    for item in gold_data:
        page_content = f"""
        --- VERIFIED CASE (ID: {item['id']}) ---
        Review: "{item['review_text']}"
        Correct Label: {json.dumps(item['verified_labels'], ensure_ascii=False)}
        Expert Explanation: {item['human_reasoning']}
        """
        
        # Metadata giúp lọc dữ liệu sau này
        meta = {
            "source": "human_verified",
            "id": item['id'],
            "difficulty": item.get("difficulty_level", "normal")
        }
        
        doc = Document(page_content=page_content, metadata=meta)
        documents.append(doc)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    print(f"Đang tạo database tại {persist_dir}...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("Đã cập nhật kho dữ liệu mẫu chuẩn.")

if __name__ == "__main__":
    build_verified_db()