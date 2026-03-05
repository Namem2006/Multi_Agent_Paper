import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API key từ file .env
load_dotenv()

class ACSARetriever:
    def __init__(self, base_dir=r"D:\Project\multi agent\work"):
        

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        guideline_path = os.path.join(base_dir, "system_data", "chroma_db")
        if os.path.exists(guideline_path):
            self.db_guideline = Chroma(persist_directory=guideline_path, embedding_function=self.embeddings)
            print("Đã load DB Guideline.")
        else:
            self.db_guideline = None
            print("Chưa tìm thấy DB Guideline.")

        gold_path = os.path.join(base_dir, "system_data", "chroma_db_verified")
        if os.path.exists(gold_path):
            self.db_gold = Chroma(persist_directory=gold_path, embedding_function=self.embeddings)
            print(" Đã load DB Verified Examples (Án lệ).")
        else:
            self.db_gold = None
            print("Chưa tìm thấy DB Verified Examples. Hãy chạy build_verified_db.py trước.")

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
    new_review = "Phòng ốc rộng rãi, sạch sẽ nhưng thái độ nhân viên lễ tân hơi kém."
    
    print(f"\n[Test RAG] Input Review: {new_review}")
    print("-" * 50)
    
    full_context = rag.get_combined_context(new_review)
    print(full_context)