"""
Debate & Summary Prompt Templates (English)

Each annotator defends their FULL label set.
Agents argue simultaneously; judge picks ONE overall winner.
"""


def create_debate_prompt_template() -> str:
    """
    Prompt template for debate agents.

    Placeholders:
        review_text, guideline_content,
        my_name, my_labels_text, opponent_name, opponent_labels_text,
        differing_summary, conversation_history, opponent_last_round,
        my_labels_json_block, differing_labels_hints, num_my_labels
    """
    template = """# ROLE
You are **{my_name}** - an Expert ACSA Annotator defending your full label set in a professional debate.
Objective: defend YOUR ENTIRE initial label set and prove the opponent's labels are less accurate.

---

# INPUT DATA

## 1. Review Text:
**"{review_text}"**

## 2. YOUR Label Set ({my_name}) - KEEP UNCHANGED THROUGHOUT:
{my_labels_text}

## 3. OPPONENT Label Set ({opponent_name}):
{opponent_labels_text}

## 4. Differing Labels (disagreements between both sides):
{differing_summary}

## 5. Relevant ACSA Guidelines (Retrieved from Knowledge Base):
{retrieved_context}

## 6. Debate History:
{conversation_history}

**ROUND INFO:**
- {opponent_name}'s last round: **Round {opponent_last_round}**
- You MUST start your opinion with: "Countering {opponent_name}'s argument in round {opponent_last_round}, ..."
- If round = 0: "Countering {opponent_name}'s initial position (round 0), ..."

---

# TASK
Defend YOUR ENTIRE label set. Focus especially on the Differing Labels in section 4.

# PROCEDURE
1. **Re-read the Review** and cross-check each of YOUR labels against the ACSA GUIDELINES.
2. **For each Differing Label**: explain why YOUR label is correct and the opponent is wrong.
3. **Single unified response**: one response covering ALL differing labels together.

# CRITICAL RULES
- **KEEP ALL initial labels UNCHANGED**: do NOT alter any entity/attribute/sentiment.
- **`labels` array MUST have exactly {num_my_labels} elements** - COPY from the block below.
- **Unified opinion**: one argument paragraph covering all differing labels.
- **Cite specific Guideline rules**: every argument must reference section number + verbatim content.
- **Language**: ENGLISH throughout.

# OUTPUT FORMAT
**LABELS YOU MUST RETURN (COPY EXACTLY - DO NOT CHANGE):**
{my_labels_json_block}

Full JSON:
{{{{
  "labels": <COPY ABOVE VALUES - exactly {num_my_labels} elements>,
  "opinion": "<Unified argument. MUST start: 'Countering {opponent_name}'s argument in round {opponent_last_round}, ...'. Then argue FOR EACH DIFFERING LABEL: why YOUR label is correct, opponent is wrong. Write fully and in detail.>",
  "evidence": "<Specific, REAL Guideline citations for each differing label:
{differing_labels_hints}>"
}}}}

**MANDATORY CONSTRAINTS:**
- `labels` array MUST have exactly {num_my_labels} elements, verbatim from the block above
- `evidence` MUST contain real Guideline content - NO [placeholder] or empty brackets
- `opinion` MUST specifically address each differing label
- Output MUST be raw JSON only (no markdown code fences, no surrounding commentary)
"""
    return template


def create_summary_prompt_template() -> str:
    """
    Prompt template for summary agent.

    Placeholders: full_response, agent_name, opponent_name, round_number, num_conflicts
    """
    template = """# ROLE
You are the **Summary Agent** - you condense debate arguments from a multi-label ACSA debate.

# TASK
Read the Full Response below carefully and produce a concise summary preserving key information:
- `labels`: COPY VERBATIM from Full Response - do NOT change any value
- `opinion`: 2-3 sentence summary, MUST start with "Countering {opponent_name}'s argument in round {round_number}, ..."
- `evidence`: Keep the 1-2 most important Guideline citations, MUST quote verbatim rule content - do NOT write [placeholder]

# INPUT - FULL RESPONSE TO SUMMARIZE
Agent: {agent_name} | Countering: {opponent_name} | Round: {round_number} | Labels count: {num_conflicts}

{full_response}

# OUTPUT FORMAT
{{
  "labels": <COPY VERBATIM the labels list from the Full Response above>,
  "opinion": "<2-3 sentences. MUST start: Countering {opponent_name}'s argument in round {round_number}, ... Summarize argument for each differing label.>",
  "evidence": "<Quote 1-2 verbatim Guideline rules from Full Response. CORRECT EXAMPLE: 'Per Guideline section 4.2: Flavor adjectives expressing satisfaction -> DRINKS#QUALITY = positive'. NEVER write [placeholder].>"
}}

**CONSTRAINTS:**
- `labels` must have exactly {num_conflicts} elements, COPIED from Full Response
- `evidence` must contain actual rule content, not placeholders
- Output MUST be raw JSON only (no markdown code fences, no surrounding commentary)
"""
    return template
