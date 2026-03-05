import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.helpers import load_prompt_from_yaml

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def summarize_debate_turn(raw_rebuttal: str):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("LỖI: Không tìm thấy GOOGLE_API_KEY")
        return None

    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    
    yaml_path = os.path.join(ROOT_DIR, "Prompt", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "summary_agent", "system_prompt")
    
    
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    
    chain = prompt | llm
    response = chain.invoke({
        "raw_rebuttal": raw_rebuttal
    })
    
    return response.content.strip()

if __name__ == "__main__":
    test_text = "Tôi không đồng ý với Annotator A. Theo mục 3 của Guideline, 'màn hình' thuộc về DISPLAY, không phải GENERAL. Hơn nữa, câu này có chữ 'nhưng' mang nghĩa phàn nàn nên Sentiment phải là NEGATIVE."
    
    print("Đang tóm tắt lập luận...")
    summary = summarize_debate_turn(test_text)
    print(f"\n[Bản tóm tắt]:\n{summary}")