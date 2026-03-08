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
        return {"error": "Invalid JSON", "raw_output": output}

def propose_guideline_update(root_cause_data: dict, current_guideline_chunk: str, target_domain: str ):
    review_id = root_cause_data.get("review_id", "Unknown")
    
    # 1. Kiem tra xem co thuc su can update luat khong
    if not root_cause_data.get("need_update", False):
        print(f"[Guideline Agent] Xung dot tai ID {review_id} khong can thiet, KHONG CAN cap nhat luat.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("BASE_URL")
    
    if not api_key or not endpoint:
        raise ValueError("[LOI] Thieu OPENAI_API_KEY hoac BASE_URL trong .env")

    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=os.getenv("API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini"),
        temperature=0.3
    )
    
    # 3. Trich xuat thong tin
    conflict_reason = root_cause_data.get("root_cause_analysis", "")
    suggestion_content = root_cause_data.get("suggestion_content", "")
    
    # 4. Goi LLM
    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "guideline_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    chain = prompt | llm
    
    print(f"\n[Guideline Agent] Dang soan thao luat moi cho ID {review_id} (Mien: {target_domain})...")
    response = chain.invoke({
        "target_domain": target_domain,
        "current_guideline": current_guideline_chunk,
        "conflict_reason": conflict_reason,
        "suggestion_content": suggestion_content
    })
    
    return clean_json_output(response.content)

def append_to_guideline_file(proposed_rule_json: dict, filepath: str):
    """
    Ghi de hoac noi them luat moi vao cuoi file adapted_guideline.txt
    """
    if not proposed_rule_json or "option_1_direct_content" not in proposed_rule_json:
        return False
        
    rule_content = proposed_rule_json["option_1_direct_content"]
    target_section = proposed_rule_json.get("target_section", "NEW RULES")
    
    formatted_rule = f"\n\n### UPDATE FOR: {target_section} ###\n{rule_content}"
    
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(formatted_rule)
        print(f"[LUU TRU] Da ghi luat moi vao file: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f"[LOI] Khong the ghi file: {e}")
        return False

if __name__ == "__main__":
    cause_file_path = os.path.join(ROOT_DIR, "system_data", "cause", "cause_data.json")
    mock_current_guideline = "- DRINKS#QUALITY: Dung cho cac mo ta ve chat luong, huong vi do uong."
    
    if os.path.exists(cause_file_path):
        with open(cause_file_path, "r", encoding="utf-8") as f:
            cause_list = json.load(f)
            
        print(f"[TEST] Tim thay {len(cause_list)} ca xung dot trong file. Dang xu ly...")
        
        for cause_data in cause_list:
            result = propose_guideline_update(cause_data, mock_current_guideline, target_domain="Restaurant")
            
            if result:
                print(f"\n[DE XUAT LUAT MOI CHO ID {cause_data.get('review_id')}]")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("-" * 50)
    else:
        print(f"[LOI] Khong tim thay file {cause_file_path}. Hay chay Root Cause Agent truoc nhe.")