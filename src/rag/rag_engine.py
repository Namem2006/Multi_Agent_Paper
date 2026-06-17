import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# CẤU HÌNH API
GOOGLE_API_KEY = "AIzaSyBrwzeGRALqrf2Hdl0s7cXnwr6QqqpOk-Q"  # Nhớ thay key của bạn
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class ACSARetriever:
    def __init__(self, base_dir=r"D:\Project\multi agent\work"):
        # 1. Khởi tạo Embedding Model (Phải khớp với lúc Build)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # 2. Kết nối DB Luật (Guideline)
        guideline_path = os.path.join(base_dir, "chroma_db_acsa")
        if os.path.exists(guideline_path):
            self.db_guideline = Chroma(persist_directory=guideline_path, embedding_function=self.embeddings)
            print("✅ [RAG] Đã load DB Guideline.")
        else:
            self.db_guideline = None
            print("⚠️ [RAG] Chưa tìm thấy DB Guideline.")

        # 3. Kết nối DB Án lệ (Gold Verified Examples) - PHẦN MỚI
        gold_path = os.path.join(base_dir, "chroma_db_verified_examples")
        if os.path.exists(gold_path):
            self.db_gold = Chroma(persist_directory=gold_path, embedding_function=self.embeddings)
            print("✅ [RAG] Đã load DB Verified Examples (Án lệ).")
        else:
            self.db_gold = None
            print("⚠️ [RAG] Chưa tìm thấy DB Verified Examples. Hãy chạy build_verified_db.py trước.")

    def retrieve_guideline(self, query, k=2):
        if not self.db_guideline: return ""
        docs = self.db_guideline.similarity_search(query, k=k)
        
        context = "### RELEVANT GUIDELINES (Luật):\n"
        for i, doc in enumerate(docs):
            context += f"- Rule {i+1}: {doc.page_content}\n"
        return context

    def retrieve_gold_examples(self, query, k=2):
        if not self.db_gold: return ""
        docs = self.db_gold.similarity_search(query, k=k)
        
        context = "### SIMILAR VERIFIED CASES:\n"
        for i, doc in enumerate(docs):
            context += f"{doc.page_content}\n"
            
        return context

    def get_combined_context(self, query):
        guideline_txt = self.retrieve_guideline(query)
        example_txt = self.retrieve_gold_examples(query)
        
        return f"{guideline_txt}\n\n{example_txt}"


if __name__ == "__main__":
    rag = ACSARetriever()
    new_review = "Mạng mẽo ở quán chán quá, quay vòng vòng không xem được Youtube."
    
    print(f"\n🔍 Input Review: {new_review}")
    print("-" * 50)
    
    # Lấy thông tin
    full_context = rag.get_combined_context(new_review)
    
    print(full_context)