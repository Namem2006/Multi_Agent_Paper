import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.helpers import load_prompt_from_yaml

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def clean_json_output(output: str):
    """Don dep markdown va parse JSON an toan"""
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
        return json.loads(output)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned by LLM", "raw_output": output}

def analyze_root_cause(debate_result: dict):
    api_key_azure = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("BASE_URL")
    api_version = os.getenv("API_VERSION")
    deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")

    if not api_key_azure or not azure_endpoint:
        raise ValueError("[LOI] Thieu cau hinh Azure OpenAI (OPENAI_API_KEY, BASE_URL) trong file .env")

    llm = AzureChatOpenAI(
        api_key=api_key_azure,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=deployment_name,
        temperature=0.2 
    )
 
    review_text = debate_result.get("review_text", "")
    
    # SUA LOI: Lay dung key "sample_id" tu file JSON, neu khong co moi lay "review_id"
    review_id = debate_result.get("sample_id", debate_result.get("review_id", "unknown_id"))
    
    debate_history = json.dumps(debate_result.get("debate_summary", {}), ensure_ascii=False, indent=2)

    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "root_cause_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    chain = prompt | llm
    
    print(f"\n[Root Cause Agent] Dang phan tich nguyen nhan xung dot cho ID: {review_id}...")
    response = chain.invoke({
        "review_text": review_text, 
        "debate_history": debate_history
    })
    
    parsed_result = clean_json_output(response.content)
    
    if isinstance(parsed_result, dict):
        parsed_result["review_id"] = review_id

    cause_dir = os.path.join(ROOT_DIR, "system_data", "cause")
    os.makedirs(cause_dir, exist_ok=True)
    master_file_path = os.path.join(cause_dir, "cause_data.json")
    
    existing_data = []
    
    if os.path.exists(master_file_path):
        try:
            with open(master_file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [] 
        except json.JSONDecodeError:
            existing_data = []
            
    existing_data.append(parsed_result)
    
    try:
        with open(master_file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"[LUU TRU] Da them phan tich vao file tong: {master_file_path}")
    except Exception as e:
        print(f"[LOI] Khong the luu file: {e}")

    return parsed_result

if __name__ == "__main__":
    test_json_path = os.path.join(ROOT_DIR, "system_data", "debate_results_multilabel.json")
    if os.path.exists(test_json_path):
        with open(test_json_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
            
        result = analyze_root_cause(sample_data)
        print("\n[KET QUA PHAN TICH NGUYEN NHAN]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Khong tim thay file test tai {test_json_path}")