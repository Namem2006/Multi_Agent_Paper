"""
Summary Agent (Moderator) for Debate System

Responsible for:
- Recording debate responses to history
- Summarizing full debate responses into concise 2-3 sentence versions
- Formatting round entries
- Managing history flow
"""

import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from models.schemas import DebateState, DebateResponse, ACSALabel
from memory_and_history.history_manager import HistoryManager
from prompts.debate_prompts import create_summary_prompt_template
from config.llm_config import get_llm


class SummaryAgent:
    """
    Moderator agent that manages debate flow and history recording
    
    The SummaryAgent acts as a neutral moderator that:
    1. Receives FULL responses from debating agents (long, detailed)
    2. Uses LLM to SUMMARIZE them into 2-3 sentence versions
    3. Formats them into structured round entries
    4. Appends SUMMARIZED versions to appropriate case history
    """
    
    def __init__(self, llm=None):
        """
        Initialize Summary Agent
        
        Args:
            llm: Optional LangChain LLM instance. If None, creates default Azure OpenAI LLM
        """
        self.history_manager = HistoryManager()
        self.llm = llm if llm is not None else get_llm()
        self.summary_prompt_template = create_summary_prompt_template()
    
    def summarize_response(self, full_response: Dict[str, Any], agent_name: str, opponent_name: str, round_number: int) -> Dict[str, Any]:
        """
        Summarize a full debate response into concise 2-3 sentence version using LLM
        
        Takes a detailed debate response and condenses it while maintaining:
        - Exact same label (entity, attribute, sentiment)
        - Core opinion starting with "Phản biện lại [opponent] ở round X trước đó"
        - Most important evidence citations
        
        Args:
            full_response: Complete debate response with long opinion and evidence
            agent_name: Name of the agent who generated this response (A1 or A2)
            opponent_name: Name of the opponent (A2 or A1)
            round_number: The round number being referenced in the response
            
        Returns:
            Summarized response with same structure but shorter content
        """
        # Build prompt for summarization
        full_response_json = json.dumps(full_response, ensure_ascii=False, indent=2)
        
        prompt = self.summary_prompt_template.format(
            full_response=full_response_json,
            agent_name=agent_name,
            opponent_name=opponent_name,
            round_number=round_number
        )
        
        # Call LLM to get summarized version
        messages = [HumanMessage(content=prompt)]
        response = self.llm.with_structured_output(DebateResponse).invoke(messages)
        
        # Convert to dict
        summarized = {
            "label": {
                "entity": response.label.entity,
                "attribute": response.label.attribute,
                "sentiment": response.label.sentiment
            },
            "opinion": response.opinion,
            "evidence": response.evidence
        }
        
        return summarized
    
    def record_response(self, state: DebateState) -> DebateState:
        """
        Record agent's response to debate history AFTER summarizing it
        
        This is the main moderator function that:
        - Retrieves the FULL response from temporary storage
        - Uses LLM to SUMMARIZE it into 2-3 sentences
        - Appends SUMMARIZED version to appropriate case history
        - Increments the round counter
        - Resets the turn marker
        
        Args:
            state: Current debate state with last_response populated
            
        Returns:
            Updated state with SUMMARIZED response recorded in history
        """
        # Get last response from temporary storage
        last_response = state.get("last_response", {})
        annotator = last_response.get("annotator")
        full_response = last_response.get("response")
        
        if not annotator or not full_response:
            print("Warning: No valid response to record")
            return state
        
        # Log the recording action
        current_case = state["current_case"]
        current_round = self.history_manager.get_current_round(state)
        print(f"  → Moderator đang tóm tắt và ghi nhận response của {annotator} (Case {current_case}, Round {current_round})...")
        
        # Determine opponent name and round to reference
        opponent_name = "A2" if annotator == "A1" else "A1"
        
        # Calculate opponent's last round based on current round
        if current_round == 1:
            opponent_last_round = 0  # First response references opponent's initial position
        else:
            opponent_last_round = current_round - 1
        
        # SUMMARIZE the full response using LLM
        summarized_response = self.summarize_response(
            full_response=full_response,
            agent_name=annotator,
            opponent_name=opponent_name,
            round_number=opponent_last_round
        )
        
        # Append SUMMARIZED version to history using HistoryManager
        state = self.history_manager.append_response(state, annotator, summarized_response)
        
        # Reset turn marker
        state["current_turn"] = None
        
        return state


# Factory function
def create_summary_agent(llm=None) -> SummaryAgent:
    """
    Factory function to create a summary agent with optional LLM
    
    Args:
        llm: Optional LLM instance. If None, will use default from config
        
    Returns:
        Configured SummaryAgent instance
    """
    return SummaryAgent(llm=llm)
