import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
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
        print(f"LỖI: Không tìm thấy file gốc tại {source_file_path}")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    yaml_path = os.path.join(ROOT_DIR, "Prompt", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "adapt_agent", "system_prompt")
    
    prompt = ChatPromptTemplate.from_template(prompt_str)
    chain = prompt | llm

    print("Đang phân tích và sinh Guideline mới qua Google AI Studio...")
    response = chain.invoke({
        "source_guideline_content": source_guideline_content,
        "target_domain_name": target_domain,
        "sample_reviews": samples
    })

    adapted_content = response.content

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
        
    print(f"--- THÀNH CÔNG! Đã lưu Guideline mới tại: {output_file_path} ---")

if __name__ == "__main__":
    sample_data = """
    1. "Ảnh chụp từ hôm qua, đi chơi với gia đình và 1 nhà họ hàng đang sống tại Sài Gòn. _ Hôm qua đi ăn trưa muộn, ai cũng đói hết nên lúc có đồ ăn là nhào vô ăn liền, bởi vậy mới quên chụp các phần gọi thêm với nước mắm, chỉ chụp món chính thôi! _ Đói quá nên không biết đánh giá đồ ăn kiểu gì luôn 😅😅😅_ Chọn cái này vì thấy nó lạ với tui."
    2. "Hương vị thơm ngon, ăn cay cay rất thích, nêm nếm vừa miệng. Ngoài ra menu quán cũng nhiều món khác nhau tha hồ cho bạn lựa chọn luôn._Quán rộng rãi, view khá đẹp và cũng thoáng lắm. Khách của quán đông nên nhiều khi nhân viên phục vụ không được nhanh cho lắm._Thịt heo rừng giá theo kì nên nhớ hỏi trước nhưng trung bình là 150k 1 phần. Hai bạn đi chung chi khoảng 400k hơn là ổn nhé."
    3. "1 bàn tiệc hoành tráng 3 đứa ăn no muốn tắt thở mà giá chỉ 228k (ăn trung đợt giảm 10%), mình thích nhất pad thái vs gà nướng - đúng kiểu Thái luôn, quán nhỏ, trong hẻm, cơ mà anh phục vụ rất dễ thương, chế biến hợp vệ sinh, bếp lộ thiên lại còn có thể xin nấu cùng - đã!"
    4. "Cháo: có nhiều hương cho các bạn chọn, nhưng mình thì chọn hương lá dứa. Cháo thì cỡ hai chén là hết, cháo khá nhừ và thơm, cháo thơm mùi gạo tự nhiên😍- Ếch: vị satế, vì mình cực kì thích ăn cay. Nồi ếch thơm lừng , ăn tại chỗ thì ếch nóng hổi , ăn nóng bao ngon luôn. Cay cay, mặn mặn hoà chung với cháo thì siêu ngon luôn😝"
    5. "Đồ nướng thì chỗ này không ít bạn "chẻ" biết đến đâu nhỉ =))) - Giá thì cực sinh viên luôn . Chỉ từ 5k 1 xiên thôi . Có cả bò phô mai với cả bạch tuộc nướng nưa- Thịt ở đây ướp rất là ngon vừa miệng ăn nữa - Nói chùn là không có chỗ chê hê hê
"
    """
    
    SOURCE_PATH = os.path.join(ROOT_DIR, "guideline.txt")
    TARGET_DOMAIN = "Restr"
    OUTPUT_PATH = os.path.join(ROOT_DIR, "adapted_guideline.txt")
    
    generate_adapted_guideline(
        source_file_path=SOURCE_PATH,
        target_domain=TARGET_DOMAIN,
        samples=sample_data,
        output_file_path=OUTPUT_PATH
    )