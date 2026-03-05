"""
Judge Prompt Template

Judge agents review the full debate history and pick ONE overall winner:
the annotator (A1 or A2) whose ENTIRE label set is more correct.
"""


def create_judge_prompt_template() -> str:
    """
    Create prompt template for judge agents.

    Judge reads both annotators' full label sets and the full debate history,
    then selects which annotator overall has the more accurate label set.

    Placeholders:
        judge_name, review_text,
        a1_labels_text, a2_labels_text,
        guideline_content, case_1_history, case_2_history
    """

    template = """# ROLE
Bạn là **{judge_name}** — Trọng tài Gán nhãn ACSA cấp cao, độc lập và không thiên vị.

Nhiệm vụ: Đọc toàn bộ bộ nhãn của A1 và A2, đọc toàn bộ lịch sử tranh luận, sau đó chọn **MỘT người chú thích** có bộ nhãn TỔNG THỂ chính xác hơn.

---

# INPUT DATA

## 1. Câu Review:
**"{review_text}"**

## 2. Bộ nhãn đầy đủ của A1:
{a1_labels_text}

## 3. Bộ nhãn đầy đủ của A2:
{a2_labels_text}

## 4. ACSA Guidelines:
{guideline_content}

## 5. Lịch sử Tranh luận — Case 1 (A1 đi trước):
{case_1_history}

## 6. Lịch sử Tranh luận — Case 2 (A2 đi trước):
{case_2_history}

---

# TASK
1. Đọc kỹ câu Review và cả 2 bộ nhãn của A1, A2.
2. Đối chiếu TỪNG nhãn với ACSA Guidelines.
3. Phân tích tranh luận để xem bên nào lập luận thuyết phục hơn.
4. Chọn **MỘT** annotator (A1 HOẶC A2) có bộ nhãn tổng thể chính xác hơn.

---

# PROCEDURE
**Bước 1** — Xác định các nhãn BẤT ĐỒNG giữa A1 và A2.

**Bước 2** — Với TỪNG nhãn bất đồng:
- Nhãn A1 có phù hợp Guideline không?
- Nhãn A2 có phù hợp Guideline không?
- Bên nào lập luận trong lịch sử tranh luận thuyết phục hơn?

**Bước 3** — Đưa ra phán quyết TỔNG THỂ:
- Xét cả bộ nhãn: bên nào đúng nhiều hơn trong các nhãn bất đồng?
- Chọn A1 hoặc A2 làm người thắng

---

# CRITICAL RULES
- **Chỉ chọn "A1" hoặc "A2"** — không đề xuất nhãn mới
- **`winner_annotator` PHẢI là "A1" hoặc "A2"** (chữ hoa, giữ nguyên)
- **Dựa vào ACSA Guidelines** — không phán quyết theo cảm tính
- **Ngôn ngữ**: reasoning và key_evidence PHẢI bằng TIẾNG VIỆT

# OUTPUT FORMAT
{{{{
  "decision": {{{{
    "winner_annotator": "<'A1' hoặc 'A2'>",
    "reasoning": "<Lý do tại sao bộ nhãn của người này chính xác hơn tổng thể. Đề cập cụ thể các nhãn bất đồng và tại sao bên thắng đúng hơn. TIẾNG VIỆT.>",
    "key_evidence": "<Trích dẫn Guideline quan trọng nhất hỗ trợ quyết định. Ví dụ: 'Theo mục 4.2: [nội dung nguyên văn]'. TIẾNG VIỆT.>"
  }}}}
}}}}

**LƯU Ý:**
- `winner_annotator` chỉ có 2 giá trị hợp lệ: "A1" hoặc "A2"
- Chỉ có 1 phán quyết duy nhất (không phải danh sách)
"""

    return template
