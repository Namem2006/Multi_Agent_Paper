"""
Pydantic schemas for ACSA Multi-Agent Debate System

Defines structured data models for:
- ACSA labels (entity, attribute, sentiment)
- Debate responses
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


class DebateResponse(BaseModel):
    """
    Structured debate response with simple structure: label, opinion, evidence
    
    Attributes:
        label: ACSA label - must maintain initial position throughout debate
        opinion: Detailed argument in Vietnamese, must reference opponent's previous round
        evidence: Must cite specific sections from Guidelines with section numbers
    """
    label: ACSALabel = Field(
        description="Nhãn ACSA - PHẢI GIỮ NGUYÊN quan điểm ban đầu của bạn, KHÔNG ĐƯỢC thay đổi"
    )
    opinion: str = Field(
        description=(
            "Lập luận tranh luận chi tiết bằng tiếng Việt. "
            "PHẢI nêu rõ 'phản biện lại {opponent_name} ở round X trước đó' nếu không phải round đầu. "
            "Giải thích tại sao quan điểm của BẠN đúng và đối thủ sai, trích dẫn Guideline."
        )
    )
    evidence: str = Field(
        description=(
            "Bằng chứng PHẢI trích dẫn CỤ THỂ từ Guideline. "
            "Ví dụ: 'Theo mục 2.1 của Guideline: [trích dẫn nguyên văn]' "
            "hoặc 'Guideline nêu rõ tại mục 3: [nội dung cụ thể]'"
        )
    )


class DebateState(TypedDict):
    """
    State management for debate graph
    
    Manages:
    - Input data (sample, text, conflict info)
    - Initial positions of both annotators
    - Two parallel debate cases (A1 first, A2 first)
    - Control flow and routing
    - Final consolidated output
    """
    # Input data
    sample_id: str
    text: str
    conflict_aspect: Dict[str, str]  # Which aspect has conflict
    
    # Initial positions
    A1_initial: Dict[str, Any]
    A2_initial: Dict[str, Any]
    
    # Case 1: A1 attacks A2's initial label
    history_case_1: List[Dict[str, Any]]
    current_case_1_round: int
    
    # Case 2: A2 attacks A1's initial label
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
