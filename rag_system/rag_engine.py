import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config.embedding_config import get_embeddings

# Load API key từ file .env
load_dotenv()

class ACSARetriever:
    def __init__(self, base_dir=r"D:\Project\multi agent\work"):

        # Sử dụng shared embedding instance
        self.embeddings = get_embeddings()

        guideline_path = os.path.join(base_dir, "system_data", "chroma_db")
        if os.path.exists(guideline_path):
            self.db_guideline = Chroma(persist_directory=guideline_path, embedding_function=self.embeddings)
            print("Đã load DB Guideline.")
        else:
            self.db_guideline = None
            print("Chưa tìm thấy DB Guideline.")

        agreed_path = os.path.join(base_dir, "system_data", "chroma_db_agreed")
        if os.path.exists(agreed_path):
            self.db_agreed = Chroma(persist_directory=agreed_path, embedding_function=self.embeddings)
            print(" Đã load DB Agreed Examples (Án lệ đồng thuận).")
        else:
            self.db_agreed = None
            print("Chưa tìm thấy DB Agreed Examples. Hãy chạy build_agreed_db.py trước.")

    def retrieve_guideline(self, query, k=2):
        if not self.db_guideline: return ""
        docs = self.db_guideline.similarity_search(query, k=k)

        context = "### RELEVANT GUIDELINES (Luật):\n"
        for i, doc in enumerate(docs):
            context += f"- Rule {i+1}: {doc.page_content}\n"
        return context

    def retrieve_agreed_examples(self, query, k=2):
        if not self.db_agreed: return ""
        docs = self.db_agreed.similarity_search(query, k=k)

        context = "### SIMILAR AGREED CASES:\n"
        for i, doc in enumerate(docs):
            context += f"{doc.page_content}\n"

        return context

    def get_combined_context(self, query):
        guideline_txt = self.retrieve_guideline(query)
        example_txt = self.retrieve_agreed_examples(query)

        return f"{guideline_txt}\n\n{example_txt}"

if __name__ == "__main__":
    rag = ACSARetriever()
    new_review = " Ngon tuyệt vời luôn"

    print(f"\n[Test RAG] Input Review: {new_review}")
    print("-" * 50)

    full_context = rag.get_combined_context(new_review)
    print(full_context)
