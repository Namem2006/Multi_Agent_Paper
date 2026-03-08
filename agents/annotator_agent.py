import os
import sys
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI, ChatOpenAI
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

def get_retrieved_context_for_batch(batch_data: list, base_db_dir: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    combined_review_text = " ".join([item["text"] for item in batch_data])
    
    guideline_db_path = os.path.join(base_db_dir, "chroma_db")
    guidelines_text = ""
    if os.path.exists(guideline_db_path):
        vector_db = Chroma(persist_directory=guideline_db_path, embedding_function=embeddings)
        docs = vector_db.similarity_search(combined_review_text, k=4) 
        guidelines_text = "\n\n".join([doc.page_content for doc in docs])
    
    verified_db_path = os.path.join(base_db_dir, "chroma_db_verified")
    verified_examples_text = ""
    if os.path.exists(verified_db_path):
        vector_db_verified = Chroma(persist_directory=verified_db_path, embedding_function=embeddings)
        docs_verified = vector_db_verified.similarity_search(combined_review_text, k=3)
        if docs_verified:
            verified_examples_text = "\n\n".join([doc.page_content for doc in docs_verified])          
    if not verified_examples_text.strip():
        verified_examples_text = "No verified examples available for this context."

    return guidelines_text, verified_examples_text

def annotate_with_deepseek(batch_text_prompt: str, retrieved_guidelines: str, verified_examples: str):
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    if not nvidia_api_key:
        raise ValueError("[LOI] Thieu NVIDIA_API_KEY trong file .env")
    llm = ChatOpenAI(
        api_key=nvidia_api_key,
        base_url=nvidia_base_url,
        model="deepseek-ai/deepseek-v3.2",
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"thinking": False}} 
    )
    
    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    response = (prompt | llm).invoke({
        "target_reviews_batch": batch_text_prompt, 
        "retrieved_guidelines": retrieved_guidelines,
        "verified_examples": verified_examples
    })
    return clean_json_output(response.content)

def annotate_with_gpt(batch_text_prompt: str, retrieved_guidelines: str, verified_examples: str):
    llm = AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("BASE_URL"),
        api_version=os.getenv("API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini"), 
        temperature=0.1
    )
    
    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    response = (prompt | llm).invoke({
        "target_reviews_batch": batch_text_prompt,
        "retrieved_guidelines": retrieved_guidelines,
        "verified_examples": verified_examples
    })
    return clean_json_output(response.content)

def process_and_verify_batch(batch_data: list, base_db_dir: str):
    api_key_gg = os.getenv("GOOGLE_API_KEY")
    api_key_azure = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("BASE_URL")
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key_gg or not api_key_azure or not azure_endpoint or not nvidia_api_key:
        raise ValueError("[LOI] Ban can khai bao du GOOGLE_API_KEY, OPENAI_API_KEY, BASE_URL, va NVIDIA_API_KEY trong file .env")

    batch_text_prompt = ""
    for item in batch_data:
        batch_text_prompt += f"Review ID: {item['id']}\nText: {item['text']}\n---\n"

    print(f"\n[RAG System] Dang lay luat va an le chung cho Batch ({len(batch_data)} cau)...")
    rules, verified_ex = get_retrieved_context_for_batch(batch_data, base_db_dir)

    print(f"[Annotator 1] DeepSeek dang gan nhan cho {len(batch_data)} cau...")
    deepseek_batch_result = annotate_with_deepseek(batch_text_prompt, rules, verified_ex)

    print(f"[Annotator 2] Azure GPT dang gan nhan cho {len(batch_data)} cau...")
    gpt_batch_result = annotate_with_gpt(batch_text_prompt, rules, verified_ex)

    if not isinstance(deepseek_batch_result, list): deepseek_batch_result = []
    if not isinstance(gpt_batch_result, list): gpt_batch_result = []

    final_batch_results = []
    
    for item in batch_data:
        rev_id = item["id"]
        rev_text = item["text"]
        
        # Loc ra nhung nhan thuoc ve cau review dang xet
        deepseek_labels_for_id = [lbl for lbl in deepseek_batch_result if lbl.get("review_id") == rev_id]
        gpt_labels_for_id = [lbl for lbl in gpt_batch_result if lbl.get("review_id") == rev_id]
        
        # Day tung cau qua bo so sanh va luu DB
        print(f"\n[Kiem tra] So sanh ket qua cho {rev_id}")
        res = filter_and_route_conflict(rev_id, rev_text, deepseek_labels_for_id, gpt_labels_for_id)
        final_batch_results.append(res)

    return final_batch_results

if __name__ == "__main__":
    base_system_dir = os.path.join(ROOT_DIR, "system_data")
    
    # Tao mot batch gom 3 cau de test
    test_batch = [
        {"id": "REV_001", "text": "Phòng ốc rộng rãi, sạch sẽ nhưng thái độ nhân viên lễ tân hơi kém."},
        {"id": "REV_002", "text": "Đồ ăn sáng cực kỳ ngon miệng, buffet đa dạng. Cảnh biển đẹp tuyệt vời."},
        {"id": "REV_003", "text": "Wifi khách sạn rất chậm, mình không thể làm việc được."}
    ]
    
    results = process_and_verify_batch(batch_data=test_batch, base_db_dir=base_system_dir)
    print("\n[KET QUA TRA VE CHO HE THONG]")
    print(json.dumps(results, indent=2, ensure_ascii=False))