import os
import sys
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.helpers import load_prompt_from_yaml
from core_engine.conflict_filter import filter_and_route_conflict

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def clean_json_output(output: str):
    output = output.strip()
    if output.startswith("```json"):
        output = output.replace("```json\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]
    elif output.startswith("```"):
        output = output.replace("```\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return [parsed]
        return parsed
    except json.JSONDecodeError:
        return []

def get_retrieved_rules(review_text: str, db_path: str):
    # Đã sửa lại thành text-embedding-004 để không bị lỗi 404 NOT FOUND
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(review_text)
    return "\n\n".join([doc.page_content for doc in docs])

def annotate_with_gemini(review_text: str, retrieved_rules: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    response = (prompt | llm).invoke({
        "target_review": review_text, 
        "retrieved_guidelines": retrieved_rules
    })
    return clean_json_output(response.content)

def annotate_with_gpt(review_text: str, retrieved_rules: str):
    # Khởi tạo AzureChatOpenAI theo cấu hình bạn yêu cầu
    llm = AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("BASE_URL"),
        api_version=os.getenv("API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini"), # Thường Azure bắt buộc khai báo tên Deployment
        temperature=0.1
    )
    
    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    response = (prompt | llm).invoke({
        "target_review": review_text, 
        "retrieved_guidelines": retrieved_rules
    })
    return clean_json_output(response.content)

def process_and_verify_review(review_text: str, db_path: str):
    # Cập nhật check biến môi trường
    api_key_gg = os.getenv("GOOGLE_API_KEY")
    api_key_azure = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("BASE_URL")
    
    if not api_key_gg or not api_key_azure or not azure_endpoint:
        raise ValueError("[LỖI] Bạn cần khai báo đủ GOOGLE_API_KEY, OPENAI_API_KEY, và BASE_URL trong file .env")

    print(f"\n[RAG System] Đang lấy luật cho câu: '{review_text}'")
    rules = get_retrieved_rules(review_text, db_path)

    print("[Annotator 1] Gemini đang gán nhãn...")
    gemini_result = annotate_with_gemini(review_text, rules)

    print("[Annotator 2] Azure GPT đang gán nhãn...")
    gpt_result = annotate_with_gpt(review_text, rules)

    result = filter_and_route_conflict(review_text, gemini_result, gpt_result)
    
    return result

if __name__ == "__main__":
    db_dir = os.path.join(ROOT_DIR, "system_data", "chroma_db")
    test_review = "Phòng ốc rộng rãi, sạch sẽ nhưng thái độ nhân viên lễ tân hơi kém."
    
    result = process_and_verify_review(test_review, db_dir)
    print("\n[KẾT QUẢ TRẢ VỀ CHO HỆ THỐNG]")
    print(json.dumps(result, indent=2, ensure_ascii=False))