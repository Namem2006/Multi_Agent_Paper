"""
Debate Agent for ACSA Annotation Conflict Resolution

Implements debate logic for two annotators (A1, A2) who:
- Maintain their initial positions throughout debate
- Reference opponent's previous round correctly
- Cite specific guideline sections as evidence
- Argue based on ACSA guidelines, not preference
"""

from typing import Dict, Any
from langchain_openai import AzureChatOpenAI

from models.schemas import DebateState, DebateResponse
from config.llm_config import get_llm, GUIDELINE_CONTENT
from prompts.debate_prompts import create_debate_prompt_template
from memory_and_history.history_manager import HistoryManager


class DebateAgent:
    """
    Debate agent that generates arguments for ACSA annotation conflicts
    
    Each agent (A1, A2):
    - Has an initial label position (entity, attribute, sentiment)
    - Must defend this position throughout all debate rounds
    - Cannot change their stance or agree with opponent
    - Must cite specific guideline sections
    - Must reference opponent's previous round number accurately
    """
    
    def __init__(self, agent_name: str, llm: AzureChatOpenAI = None):
        """
        Initialize debate agent
        
        Args:
            agent_name: "A1" or "A2"
            llm: Optional LLM instance (defaults to configured Azure OpenAI)
        """
        self.agent_name = agent_name
        self.llm = llm if llm else get_llm()
        self.history_manager = HistoryManager()
        self.prompt_template = create_debate_prompt_template()
    
    def generate_response(self, state: DebateState) -> DebateState:
        """
        Generate debate response for current round
        
        Process:
        1. Build debate history context
        2. Identify opponent and their last round
        3. Fill prompt template with all context
        4. Get structured response from LLM
        5. Store response in state for moderator
        
        Args:
            state: Current debate state
            
        Returns:
            Updated state with response in last_response field
        """
        # Get current case info
        current_case = state["current_case"]
        current_round = self.history_manager.get_current_round(state)
        
        # Log action
        print(f"  → {self.agent_name} đang tranh luận (Case {current_case}, Round {current_round})...")
        
        # Build prompt with full context
        prompt = self._build_prompt(state)
        
        # Get structured response from LLM
        structured_llm = self.llm.with_structured_output(DebateResponse)
        response = structured_llm.invoke([
            {"role": "user", "content": prompt}
        ])
        
        # Store in state for moderator to record
        state["last_response"] = {
            "annotator": self.agent_name,
            "response": response.model_dump()
        }
        state["current_turn"] = "moderator"
        
        return state
    
    def _build_prompt(self, state: DebateState) -> str:
        """
        Build complete prompt with all context
        
        Args:
            state: Current debate state
            
        Returns:
            Formatted prompt string ready for LLM
        """
        # Determine opponent
        opponent = "A2" if self.agent_name == "A1" else "A1"
        
        # Get my position and opponent's position
        my_label = state["A1_initial"]["label"] if self.agent_name == "A1" else state["A2_initial"]["label"]
        opponent_label = state["A2_initial"]["label"] if self.agent_name == "A1" else state["A1_initial"]["label"]
        
        # Build history text and get opponent's last round
        history_text, opponent_last_round = self.history_manager.build_history_text(state, self.agent_name)
        
        # Fill template
        prompt = self.prompt_template.format(
            review_text=state["text"],
            conflict_entity=state["conflict_aspect"]["entity"],
            conflict_attribute=state["conflict_aspect"]["attribute"],
            guideline_content=GUIDELINE_CONTENT,
            my_name=self.agent_name,
            my_entity=my_label["entity"],
            my_attribute=my_label["attribute"],
            my_sentiment=my_label["sentiment"],
            opponent_name=opponent,
            opponent_entity=opponent_label["entity"],
            opponent_attribute=opponent_label["attribute"],
            opponent_sentiment=opponent_label["sentiment"],
            conversation_history=history_text,
            opponent_last_round=opponent_last_round
        )
        
        return prompt


def create_debate_agents(llm=None) -> Dict[str, DebateAgent]:
    """
    Factory function to create both debate agents
    
    Args:
        llm: Optional shared LLM instance. If None, each agent creates its own
    
    Returns:
        Dictionary with "A1" and "A2" agents
    """
    return {
        "A1": DebateAgent("A1", llm=llm),
        "A2": DebateAgent("A2", llm=llm)
    }
