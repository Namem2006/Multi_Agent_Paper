import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.helpers import load_prompt_from_yaml

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def get_annotator_response(sample_text: str):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("LỖI: Không tìm thấy GOOGLE_API_KEY")
        return None

    db_path = os.path.join(ROOT_DIR, "chroma_db_acsa")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(sample_text)
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    yaml_path = os.path.join(ROOT_DIR, "Prompt", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    chain = prompt | llm
    
    response = chain.invoke({
        "target_review": sample_text,
        "retrieved_guidelines": retrieved_rules
    })
    
    output = response.content.strip()

    if output.startswith("```json"):
        output = output.replace("```json\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]
    elif output.startswith("```"):
        output = output.replace("```\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]

    return output

if __name__ == "__main__":
    sample = "Máy này pin quá hẻo, chơi game tí là sập nguồn."
    result = get_annotator_response(sample)
    print(result)