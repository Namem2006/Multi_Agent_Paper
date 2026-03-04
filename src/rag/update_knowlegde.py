import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

GOOGLE_API_KEY = "AIzaSyBrwzeGRALqrf2Hdl0s7cXnwr6QqqpOk-Q"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def add_new_verified_case(review_text, final_label, reasoning, human_approver="Admin"):
    persist_dir = r"D:\Project\multi agent\work\RAG\chroma_db_verified_examples"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 1. Kết nối DB hiện có
    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # 2. Tạo Document mới
    new_content = f"""
    --- VERIFIED CASE (NEWLY ADDED) ---
    Review: "{review_text}"
    Correct Label: {json.dumps(final_label, ensure_ascii=False)}
    Expert Explanation: {reasoning} (Approved by {human_approver})
    """
    
    new_doc = Document(
        page_content=new_content,
        metadata={"source": "dynamic_update", "approver": human_approver}
    )
    
    # 3. Thêm vào DB 
    vector_db.add_documents([new_doc])
    
    print(f"✅ Đã thêm case mới: '{review_text[:30]}...' vào kho tri thức.")

if __name__ == "__main__":
    # Giả sử đây là case vừa được tranh luận xong và con người chốt
    add_new_verified_case(
        review_text="Quán này view đẹp nhưng ồn ào quá.",
        final_label=[{"entity": "AMBIENCE", "attribute": "GENERAL", "sentiment": "MIXED"}],
        reasoning="Có 2 ý đối lập về không gian: đẹp (POS) và ồn (NEG) -> Mixed hoặc tách 2 nhãn."
    )