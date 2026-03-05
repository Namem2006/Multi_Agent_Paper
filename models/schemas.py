"""
Pydantic schemas for ACSA Multi-Agent Debate System

Defines structured data models for:
- ACSA labels (entity, attribute, sentiment)
- Debate responses (supports multiple conflicting labels at once)
- Judge decisions (one per conflict label)
- Debate state management
"""

from typing import TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class ACSALabel(BaseModel):
    """
    ACSA Label structure

    Attributes:
        entity: Entity category (FOOD, DRINKS, SERVICE, AMBIENCE, LOCATION, RESTAURANT)
        attribute: Attribute type (QUALITY, PRICES, STYLE&OPTIONS, GENERAL, MISCELLANEOUS)
        sentiment: Sentiment polarity (positive, negative, neutral)
    """
    entity: str = Field(
        description="Entity category (FOOD, DRINKS, SERVICE, AMBIENCE, LOCATION, RESTAURANT)"
    )
    attribute: str = Field(
        description="Attribute (QUALITY, PRICES, STYLE&OPTIONS, GENERAL, MISCELLANEOUS)"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment polarity"
    )


class MultiLabelDebateResponse(BaseModel):
    """
    Structured debate response supporting MULTIPLE conflicting labels at once.

    When a sentence has N conflicting labels, one agent defends all N of their
    positions in a single response, arguing about all of them together.

    Attributes:
        labels: List of ACSA labels — one per conflict. Order matches conflict_labels list.
                MUST keep the SAME entity/attribute/sentiment as the agent's initial positions.
        opinion: Detailed argument in Vietnamese covering ALL conflicts at once.
                 Must start with "Phản biện lại {opponent} ở round X trước đó, ..."
        evidence: Specific Guideline citations supporting all label decisions.
    """
    labels: List[ACSALabel] = Field(
        description=(
            "Danh sách nhãn ACSA — mỗi phần tử tương ứng với một conflict label. "
            "PHẢI GIỮ NGUYÊN tất cả nhãn ban đầu của bạn, KHÔNG ĐƯỢC thay đổi bất kỳ nhãn nào."
        )
    )
    opinion: str = Field(
        description=(
            "Lập luận tổng hợp bằng tiếng Việt cho TẤT CẢ các nhãn conflict. "
            "PHẢI bắt đầu: 'Phản biện lại {opponent_name} ở round X trước đó, ...' "
            "Giải thích từng nhãn tại sao quan điểm của BẠN đúng và đối thủ sai."
        )
    )
    evidence: str = Field(
        description=(
            "Bằng chứng trích dẫn CỤ THỂ từ Guideline cho tất cả các nhãn. "
            "Ví dụ: 'Theo mục 2.1: [nội dung]'"
        )
    )


class JudgeDecision(BaseModel):
    """
    A single judge's overall decision: which annotator (A1 or A2) has the
    MORE CORRECT full label set for this review.

    Attributes:
        winner_annotator: "A1" or "A2" — whose full label set is more correct
        reasoning: Detailed explanation in Vietnamese
        key_evidence: Most important guideline citation supporting the decision
    """
    winner_annotator: Literal["A1", "A2"] = Field(
        description=(
            "Người chú thích thắng: 'A1' hoặc 'A2'. "
            "Chọn người có bộ nhãn TỔNG THỂ chính xác hơn so với ACSA Guidelines."
        )
    )
    reasoning: str = Field(
        description=(
            "Lý do chi tiết bằng tiếng Việt tại sao bộ nhãn của người này chính xác hơn. "
            "Phân tích cả 2 phía và chỉ rõ bên nào đúng hơn dựa trên Guideline."
        )
    )
    key_evidence: str = Field(
        description=(
            "Trích dẫn Guideline QUAN TRỌNG NHẤT hỗ trợ quyết định. "
            "Ví dụ: 'Theo mục 4.2: [nội dung nguyên văn]'"
        )
    )


class MultiLabelJudgeResponse(BaseModel):
    """
    A single judge's complete response: one overall winner decision.

    Attributes:
        decision: The judge's single decision picking the better annotator overall
    """
    decision: JudgeDecision = Field(
        description=(
            "Phán quyết duy nhất: chọn A1 hoặc A2 có bộ nhãn tổng thể chính xác hơn."
        )
    )


class DebateState(TypedDict):
    """
    State management for multi-label debate graph.

    Supports debating MULTIPLE conflicting labels at once in a single session.
    """
    # Input data
    sample_id: str
    text: str


    # Initial positions (full label lists from each annotator)
    A1_initial: Dict[str, Any]   # {"labels": [...], "opinion": str, "evidence": str}
    A2_initial: Dict[str, Any]   # {"labels": [...], "opinion": str, "evidence": str}

    # Case 1: A1 attacks A2's positions
    history_case_1: List[Dict[str, Any]]
    current_case_1_round: int

    # Case 2: A2 attacks A1's positions
    history_case_2: List[Dict[str, Any]]
    current_case_2_round: int

    # Control flow
    max_rounds: int
    current_case: Literal["case_1", "case_2", "completed"]
    current_turn: Literal["A1", "A2", "moderator"]

    # Temporary storage for passing response to moderator
    last_response: Dict[str, Any]

    # Final output for judgment phase
    final_output: Dict[str, Any]
