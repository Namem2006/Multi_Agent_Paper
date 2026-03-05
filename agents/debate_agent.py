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

from models.schemas import DebateState, MultiLabelDebateResponse
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
        structured_llm = self.llm.with_structured_output(MultiLabelDebateResponse)
        response = structured_llm.invoke([
            {"role": "user", "content": prompt}
        ])

        # Store in state for moderator: convert labels list to list of dicts
        state["last_response"] = {
            "annotator": self.agent_name,
            "response": {
                "labels": [lbl.model_dump() for lbl in response.labels],
                "opinion": response.opinion,
                "evidence": response.evidence
            }
        }
        state["current_turn"] = "moderator"
        
        return state
    
    def _build_prompt(self, state: DebateState) -> str:
        """
        Build complete prompt for full-label-set debate.
        Agents defend ALL their labels; focus on differing labels.
        """
        opponent = "A2" if self.agent_name == "A1" else "A1"

        my_initial  = state["A1_initial"] if self.agent_name == "A1" else state["A2_initial"]
        opp_initial = state["A2_initial"] if self.agent_name == "A1" else state["A1_initial"]

        my_labels  = my_initial["labels"]   # full label list
        opp_labels = opp_initial["labels"]  # full label list

        # ── Format MY full label list ───────────────────────────────────
        my_labels_list = []
        for i, lbl in enumerate(my_labels, start=1):
            my_labels_list.append(
                f"  {i}. {lbl['entity']}#{lbl['attribute']} | {lbl['sentiment']}"
            )
        my_labels_text = "\n".join(my_labels_list)

        # ── Format OPPONENT full label list ────────────────────────────
        opp_labels_list = []
        for i, lbl in enumerate(opp_labels, start=1):
            opp_labels_list.append(
                f"  {i}. {lbl['entity']}#{lbl['attribute']} | {lbl['sentiment']}"
            )
        opponent_labels_text = "\n".join(opp_labels_list)

        # ── Build differing summary (labels that differ between A1 & A2) ─
        my_map  = {(l["entity"], l["attribute"]): l["sentiment"] for l in my_labels}
        opp_map = {(l["entity"], l["attribute"]): l["sentiment"] for l in opp_labels}

        diff_lines = []
        for (ent, attr), my_sent in my_map.items():
            opp_sent = opp_map.get((ent, attr))
            if opp_sent is None:
                diff_lines.append(
                    f"  - {ent}#{attr}: tôi={my_sent}, đối thủ KHÔNG GÁN nhãn này"
                )
            elif opp_sent != my_sent:
                diff_lines.append(
                    f"  - {ent}#{attr}: tôi={my_sent} vs đối thủ={opp_sent}"
                )
        for (ent, attr), opp_sent in opp_map.items():
            if (ent, attr) not in my_map:
                diff_lines.append(
                    f"  - {ent}#{attr}: tôi KHÔNG GÁN, đối thủ={opp_sent}"
                )
        differing_summary = "\n".join(diff_lines) if diff_lines else "  (Không có nhãn bất đồng)"

        # ── Hardcoded JSON block for OUTPUT FORMAT ──────────────────────
        import json as _json
        my_labels_json_block = _json.dumps(
            [{"entity": l["entity"], "attribute": l["attribute"], "sentiment": l["sentiment"]}
             for l in my_labels],
            ensure_ascii=False, indent=2
        )

        # ── Dynamic evidence hints for each differing label ─────────────
        hint_lines = []
        for (ent, attr), my_sent in my_map.items():
            opp_sent = opp_map.get((ent, attr))
            if opp_sent is None:
                hint_lines.append(
                    f"  Về {ent}#{attr}|{my_sent} (đối thủ không gán): "
                    f"Theo Guideline [...nguyên văn rule...] — nhãn của tôi đúng vì [...]."
                )
            elif opp_sent != my_sent:
                hint_lines.append(
                    f"  Về {ent}#{attr} (tôi={my_sent}, đối thủ={opp_sent}): "
                    f"Theo Guideline [...nguyên văn rule...] — nhãn của tôi đúng vì [...]."
                )
        for (ent, attr), opp_sent in opp_map.items():
            if (ent, attr) not in my_map:
                hint_lines.append(
                    f"  Về {ent}#{attr}|{opp_sent} (đối thủ gán, tôi không): "
                    f"Theo Guideline [...nguyên văn rule...] — khía cạnh này [không xuất hiện / không đáng gán] vì [...]."
                )
        if not hint_lines:
            hint_lines = ["  (Không có nhãn bất đồng — phân tích chung về tính chính xác)"]
        hint_lines.append(
            "  TUYỆT ĐỐI KHÔNG để [nội dung] hay placeholder trống — phải điền nội dung thực tế."
        )
        differing_labels_hints = "\n".join(hint_lines)

        # ── Build history text ──────────────────────────────────────────
        history_text, opponent_last_round = self.history_manager.build_history_text(
            state, self.agent_name
        )

        prompt = self.prompt_template.format(
            review_text=state["text"],
            guideline_content=GUIDELINE_CONTENT,
            my_name=self.agent_name,
            my_labels_text=my_labels_text,
            opponent_name=opponent,
            opponent_labels_text=opponent_labels_text,
            differing_summary=differing_summary,
            conversation_history=history_text,
            opponent_last_round=opponent_last_round,
            my_labels_json_block=my_labels_json_block,
            differing_labels_hints=differing_labels_hints,
            num_my_labels=len(my_labels),
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
