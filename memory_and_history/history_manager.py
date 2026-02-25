"""
History Manager for Debate System

Manages debate history for two parallel cases:
- Case 1: A1 initiates debate against A2's position
- Case 2: A2 initiates debate against A1's position

Each case maintains its own history with round-by-round tracking
"""

from typing import Dict, List, Any
from models.schemas import DebateState


class HistoryManager:
    """
    Manages debate history and state transitions
    
    Responsibilities:
    - Initialize debate histories for both cases
    - Append responses to appropriate case history
    - Build history text for agent prompts
    - Track opponent's last round for correct references
    - Merge final results from both cases
    """
    
    @staticmethod
    def initialize_histories(
        sample_id: str,
        text: str,
        conflict_aspect: Dict[str, str],
        A1_initial: Dict[str, Any],
        A2_initial: Dict[str, Any],
        max_rounds: int = 4
    ) -> DebateState:
        """
        Initialize debate state with empty histories
        
        Args:
            sample_id: Unique identifier for the sample
            text: Review text to debate
            conflict_aspect: Which aspect has annotation conflict
            A1_initial: A1's initial position (label, opinion, evidence)
            A2_initial: A2's initial position (label, opinion, evidence)
            max_rounds: Maximum debate rounds per case (default: 4)
            
        Returns:
            Initialized DebateState with round 0 for both cases
        """
        state = {
            # Input data
            "sample_id": sample_id,
            "text": text,
            "conflict_aspect": conflict_aspect,
            
            # Initial positions
            "A1_initial": A1_initial,
            "A2_initial": A2_initial,
            
            # Case 1: A1 attacks A2 (Round 0 = A2's initial position)
            "history_case_1": [
                {
                    "round": 0,
                    "A2": {
                        "label": A2_initial["label"],
                        "opinion": A2_initial["opinion"],
                        "evidence": A2_initial["evidence"]
                    }
                }
            ],
            "current_case_1_round": 1,
            
            # Case 2: A2 attacks A1 (Round 0 = A1's initial position)
            "history_case_2": [
                {
                    "round": 0,
                    "A1": {
                        "label": A1_initial["label"],
                        "opinion": A1_initial["opinion"],
                        "evidence": A1_initial["evidence"]
                    }
                }
            ],
            "current_case_2_round": 1,
            
            # Control flow
            "max_rounds": max_rounds,
            "current_case": "case_1",
            "current_turn": "A1",
            
            # Temporary storage
            "last_response": {},
            
            # Final output
            "final_output": {}
        }
        
        return state
    
    @staticmethod
    def append_response(
        state: DebateState,
        annotator: str,
        response: Dict[str, Any]
    ) -> DebateState:
        """
        Append annotator's response to appropriate case history
        
        Args:
            state: Current debate state
            annotator: "A1" or "A2"
            response: Debate response dict with label, opinion, evidence
            
        Returns:
            Updated state with response appended to history
        """
        # Determine which case history to update
        if state["current_case"] == "case_1":
            current_round = state["current_case_1_round"]
            
            # Create round entry
            round_entry = {
                "round": current_round,
                annotator: response
            }
            
            # Append to history
            state["history_case_1"].append(round_entry)
            state["current_case_1_round"] += 1
            
        else:  # case_2
            current_round = state["current_case_2_round"]
            
            # Create round entry
            round_entry = {
                "round": current_round,
                annotator: response
            }
            
            # Append to history
            state["history_case_2"].append(round_entry)
            state["current_case_2_round"] += 1
        
        return state
    
    @staticmethod
    def build_history_text(
        state: DebateState,
        annotator: str
    ) -> tuple[str, int]:
        """
        Build history text for agent prompt and find opponent's last round
        
        Args:
            state: Current debate state
            annotator: "A1" or "A2" - which agent needs the history
            
        Returns:
            Tuple of (history_text, opponent_last_round)
        """
        # Determine which case and opponent
        if state["current_case"] == "case_1":
            history = state["history_case_1"]
            opponent = "A2" if annotator == "A1" else "A1"
        else:  # case_2
            history = state["history_case_2"]
            opponent = "A1" if annotator == "A2" else "A2"
        
        # Build history text and find opponent's last round
        history_text = ""
        opponent_last_round = 0  # Default to 0 (initial position)
        
        if len(history) > 1:  # Skip round 0 (initial)
            history_text = "Lịch sử tranh luận:\n"
            for round_data in history[1:]:  # Start from round 1
                round_num = round_data["round"]
                if "A1" in round_data:
                    speaker = "A1"
                    response = round_data["A1"]
                    if speaker == opponent:
                        opponent_last_round = round_num
                else:
                    speaker = "A2"
                    response = round_data["A2"]
                    if speaker == opponent:
                        opponent_last_round = round_num
                
                history_text += f"\n--- Vòng {round_num} - {speaker} ---\n"
                history_text += f"Opinion: {response['opinion']}\n"
                history_text += f"Evidence: {response['evidence']}\n"
                history_text += f"Label: {response['label']['entity']}#{response['label']['attribute']}, {response['label']['sentiment']}\n"
        else:
            history_text = "(Chưa có lịch sử tranh luận. Đây là lượt đầu tiên.)"
        
        return history_text, opponent_last_round
    
    @staticmethod
    def merge_results(state: DebateState) -> Dict[str, Any]:
        """
        Merge both case histories into final output for judgment phase
        
        Args:
            state: Final debate state after both cases completed
            
        Returns:
            Structured output with both case histories
        """
        final_output = {
            "sample_id": state["sample_id"],
            "review_text": state["text"],
            "initial_positions": {
                "A1": state["A1_initial"],
                "A2": state["A2_initial"]
            },
            "debate_summary": {
                "case_1": {
                    "description": "A1 đi trước",
                    "history": state["history_case_1"]
                },
                "case_2": {
                    "description": "A2 đi trước",
                    "history": state["history_case_2"]
                }
            }
        }
        
        return final_output
    
    @staticmethod
    def get_current_round(state: DebateState) -> int:
        """
        Get current round number based on active case
        
        Args:
            state: Current debate state
            
        Returns:
            Current round number
        """
        if state["current_case"] == "case_1":
            return state["current_case_1_round"]
        else:
            return state["current_case_2_round"]
