import os
import sys
from dotenv import load_dotenv
# THAY ĐỔI: Import AzureChatOpenAI thay vì ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.helpers import load_prompt_from_yaml

load_dotenv(os.path.join(ROOT_DIR, ".env"))

def generate_adapted_guideline(source_file_path: str, target_domain: str, samples: str, output_file_path: str):
    try:
        with open(source_file_path, "r", encoding="utf-8") as f:
            source_guideline_content = f.read()
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file gốc tại {source_file_path}")
        return

    # THAY ĐỔI: Khởi tạo LLM bằng Azure OpenAI
    api_key_azure = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("BASE_URL")
    
    if not api_key_azure or not azure_endpoint:
        raise ValueError("[LỖI] Bạn cần khai báo đủ OPENAI_API_KEY và BASE_URL trong file .env")

    llm = AzureChatOpenAI(
        api_key=api_key_azure,
        azure_endpoint=azure_endpoint,
        api_version=os.getenv("API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini"), # Đảm bảo tên Deployment đúng
        temperature=0.1
    )

    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "adapt_agent", "system_prompt")
    
    prompt = ChatPromptTemplate.from_template(prompt_str)
    chain = prompt | llm
    
    print(f"[Adapt Agent] Đang dùng GPT phân tích dữ liệu và tạo luật cho domain '{target_domain}'...")
    response = chain.invoke({
        "source_guideline_content": source_guideline_content,
        "target_domain_name": target_domain,
        "sample_reviews": samples
    })

    adapted_content = response.content

    # Làm sạch markdown output nếu có
    if adapted_content.startswith("```markdown"):
        adapted_content = adapted_content.replace("```markdown\n", "", 1)
        if adapted_content.endswith("```"):
            adapted_content = adapted_content[:-3]
    elif adapted_content.startswith("```"):
        adapted_content = adapted_content.replace("```\n", "", 1)
        if adapted_content.endswith("```"):
            adapted_content = adapted_content[:-3]

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(adapted_content.strip())
        
    print(f"[THÀNH CÔNG] Đã lưu Guideline mới tại: {output_file_path} ---")

if __name__ == "__main__":
    sample_data = """
    1. "Phòng ốc rộng rãi, sạch sẽ, có cửa sổ nhìn ra biển rất đẹp. Nhân viên lễ tân nhiệt tình thân thiện, hỗ trợ check-in sớm không tính phí."
    2. "Khách sạn nằm ngay trung tâm, đi bộ ra chợ đêm rất gần. Tuy nhiên cách âm phòng hơi kém, ban đêm ngủ nghe tiếng ồn từ hành lang. Giá cả hợp lý so với vị trí."
    3. "Buffet sáng hơi nghèo nàn, ít món và đồ ăn nguội. Bù lại hồ bơi trên sân thượng view siêu đỉnh, nước sạch. Sẽ quay lại vì cái hồ bơi này."
    4. "Trải nghiệm tệ! Vòi sen trong phòng tắm bị rỉ nước, gọi nhân viên bảo trì lên sửa thì đợi gần 1 tiếng mới có người lên. Giường ngủ thì cứng đau hết cả lưng."
    5. "Giá phòng đợt lễ có tăng chút đỉnh nhưng chấp nhận được. Không gian yên tĩnh, thích hợp để nghỉ dưỡng. Wifi hơi chập chờn lúc buổi tối."
    """
    
    SOURCE_PATH = os.path.join(ROOT_DIR, "guideline.txt")
    TARGET_DOMAIN = "Hotel"
    OUTPUT_PATH = os.path.join(ROOT_DIR, "adapted_guideline.txt")
    
    generate_adapted_guideline(
        source_file_path=SOURCE_PATH,
        target_domain=TARGET_DOMAIN,
        samples=sample_data,
        output_file_path=OUTPUT_PATH
    )