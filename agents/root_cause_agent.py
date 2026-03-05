from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.helpers import load_prompt_from_yaml
def analyze_root_cause(sample_text: str, final_history: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    prompt_str = load_prompt_from_yaml("agent_prompt.yaml", "root_cause_agent", "system_prompt")
    
    chain = prompt_str | llm
    return chain.invoke({"text": sample_text, "history": final_history}).content