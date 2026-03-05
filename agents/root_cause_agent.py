import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.helpers import load_prompt_from_yaml

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def analyze_root_cause(sample_text: str, final_history: str, current_guideline_content: str):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("LỖI: Không tìm thấy GOOGLE_API_KEY")
        return None

    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    
    # Đọc cấu hình từ file YAML
    yaml_path = os.path.join(ROOT_DIR, "prompt", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "root_cause_agent", "system_prompt")
    
    
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    chain = prompt | llm
    
    
    response = chain.invoke({
        "conflict_sample": sample_text,
        "debate_and_verdict": final_history,
        "current_guideline_content": current_guideline_content
    })
    
    # Clean output nếu LLM trả về markdown JSON block
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