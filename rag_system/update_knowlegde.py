import json
import os
import sys
from langchain_chroma import Chroma
from langchain_core.documents import Document

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config.embedding_config import get_embeddings

def add_new_agreed_case(review_text, final_label, reasoning, human_approver="Admin"):
    persist_dir = r""
    # Sử dụng shared embedding instance
    embeddings = get_embeddings()


    vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)


    new_content = f"""
    --- AGREED CASE (NEWLY ADDED) ---
    Review: "{review_text}"
    Agreed Label: {json.dumps(final_label, ensure_ascii=False)}
    Expert Explanation: {reasoning} (Approved by {human_approver})
    """

    new_doc = Document(
        page_content=new_content,
        metadata={"source": "dynamic_update", "approver": human_approver}
    )


    vector_db.add_documents([new_doc])

    print(f"Đã thêm case mới: '{review_text[:30]}...' vào kho tri thức.")

if __name__ == "__main__":
    add_new_agreed_case(
        review_text="Quán này view đẹp nhưng ồn ào quá.",
        final_label=[{"entity": "AMBIENCE", "attribute": "GENERAL", "sentiment": "MIXED"}],
        reasoning="Có 2 ý đối lập về không gian: đẹp (POS) và ồn (NEG) -> Mixed hoặc tách 2 nhãn."
    )
