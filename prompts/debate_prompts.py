"""
Debate & Summary Prompt Templates

Each annotator defends their FULL label set (not individual conflict pairs).
Agents argue about ALL labels simultaneously; judge picks ONE overall winner.
"""


def create_debate_prompt_template() -> str:
    """
    Create prompt template for debate agents.

    Annotators defend ALL their labels against the opponent's full label set.
    Focus on labels that DIFFER from the opponent.

    Placeholders:
        review_text, guideline_content,
        my_name, my_labels_text, opponent_name, opponent_labels_text,
        differing_summary, conversation_history, opponent_last_round,
        my_labels_json_block, differing_labels_hints, num_my_labels
    """

    template = """# ROLE
Bạn là **{my_name}** — Chuyên gia Gán nhãn ACSA đang tranh luận bảo vệ bộ nhãn của mình.
Mục tiêu: bảo vệ **TOÀN BỘ** bộ nhãn ban đầu của bạn và chứng minh bộ nhãn của đối thủ kém chính xác hơn.

---

# INPUT DATA

## 1. Câu Review:
**"{review_text}"**

## 2. Bộ nhãn của BẠN ({my_name}) — GIỮ NGUYÊN ĐẾN HẾT:
{my_labels_text}

## 3. Bộ nhãn của ĐỐI THỦ ({opponent_name}):
{opponent_labels_text}

## 4. Nhãn BẤT ĐỒNG (khác nhau giữa 2 bên):
{differing_summary}

## 5. ACSA GUIDELINES (Luật chuẩn):
{guideline_content}

## 6. Lịch sử Tranh luận:
{conversation_history}

**THÔNG TIN ROUND:**
- Round cuối cùng của {opponent_name}: **Round {opponent_last_round}**
- Bạn PHẢI bắt đầu opinion: "Phản biện lại {opponent_name} ở round {opponent_last_round} trước đó, ..."
- Nếu round = 0: "Phản biện lại {opponent_name} ở round 0 (quan điểm ban đầu), ..."

---

# TASK
Bảo vệ **TOÀN BỘ** bộ nhãn của bạn. Tập trung đặc biệt vào các nhãn BẤT ĐỒNG ở mục 4.

# PROCEDURE
1. **Đọc lại câu Review** và đối chiếu từng nhãn của BẠN với ACSA GUIDELINES.
2. **Với mỗi nhãn BẤT ĐỒNG**: chỉ ra tại sao nhãn của bạn đúng và đối thủ sai.
3. **Trả lời tổng hợp**: Một response duy nhất bao gồm lập luận cho TẤT CẢ nhãn bất đồng.

# CRITICAL RULES
- **GIỮ NGUYÊN TOÀN BỘ nhãn ban đầu**: KHÔNG thay đổi bất kỳ entity/attribute/sentiment nào.
- **Mảng `labels` PHẢI có đúng {num_my_labels} phần tử** — COPY từ bảng dưới.
- **Opinion tổng hợp**: Lập luận bao quát tất cả nhãn bất đồng.
- **Trích dẫn Guideline cụ thể**: Mọi lập luận phải có số mục + nội dung nguyên văn.
- **Ngôn ngữ**: TIẾNG VIỆT toàn bộ.

# OUTPUT FORMAT
**LABELS BẠN PHẢI TRẢ VỀ (COPY NGUYÊN XI, KHÔNG ĐỔI GÌ):**
{my_labels_json_block}

JSON đầy đủ:
{{{{
  "labels": <COPY NGUYÊN GIÁ TRỊ TỪ BẢNG TRÊN — đúng {num_my_labels} phần tử>,
  "opinion": "<Lập luận tổng hợp. BẮT BUỘC bắt đầu: 'Phản biện lại {opponent_name} ở round {opponent_last_round} trước đó, ...'. Sau đó lập luận CHO TỪNG NHÃN BẤT ĐỒNG: tại sao nhãn của BẠN đúng, đối thủ sai. VIẾT ĐẦY ĐỦ, chi tiết.>",
  "evidence": "<Trích dẫn Guideline THỰC TẾ và CỤ THỂ cho TỪNG nhãn bất đồng theo format:
{differing_labels_hints}>"
}}}}

**RÀNG BUỘC BẮT BUỘC:**
- Mảng `labels` PHẢI có đúng {num_my_labels} phần tử, COPY nguyên xi từ bảng trên
- `evidence` PHẢI chứa nội dung Guideline thực tế, KHÔNG dùng [nội dung] hay placeholder
- `opinion` PHẢI đề cập từng nhãn bất đồng một cách cụ thể
"""

    return template


def create_summary_prompt_template() -> str:
    """
    Create prompt template for summary agent (multi-label support).

    Placeholders: full_response, agent_name, opponent_name, round_number, num_conflicts
    """

    template = """# ROLE
Bạn là **Summary Agent** — tóm tắt lập luận từ cuộc tranh luận ACSA đa nhãn.

# TASK
Đọc kỹ Full Response bên dưới rồi tóm tắt thành ngắn gọn hơn, giữ đủ thông tin:
- `labels`: COPY NGUYÊN XI từ Full Response — KHÔNG thay đổi bất kỳ giá trị nào
- `opinion`: 2-3 câu tóm tắt, BẮT BUỘC bắt đầu "Phản biện lại {opponent_name} ở round {round_number} trước đó, ..."
- `evidence`: Giữ lại 1-2 trích dẫn Guideline quan trọng nhất, PHẢI trích nguyên văn nội dung rule — KHÔNG viết [nội dung] hay để trống

# INPUT — FULL RESPONSE CẦN TÓM TẮT
Agent: {agent_name} | Đang phản biện: {opponent_name} | Round: {round_number} | Số conflict: {num_conflicts}

{full_response}

# OUTPUT FORMAT
{{
  "labels": <COPY NGUYÊN XI danh sách labels từ Full Response trên>,
  "opinion": "<2-3 câu. BẮT BUỘC bắt đầu: Phản biện lại {opponent_name} ở round {round_number} trước đó, ... Tóm tắt lập luận cho từng conflict>",
  "evidence": "<Trích nguyên văn 1-2 rule từ Guideline trong Full Response. VÍ DỤ ĐÚNG: 'Theo Guideline mục 4.2: Tính từ mô tả hương vị kèm cảm xúc hài lòng → DRINKS#QUALITY = positive'. TUYỆT ĐỐI KHÔNG viết dạng [nội dung]>"
}}

**RÀNG BUỘC:**
- `labels` phải có đúng {num_conflicts} phần tử, COPY từ Full Response
- `evidence` phải chứa nội dung rule thực tế, không phải placeholder
"""

    return template
