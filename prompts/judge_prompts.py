"""Prompt templates for primary judges and tie-break judge (English)."""


def _shared_confidence_rules() -> str:
    return """# CONFIDENCE CALIBRATION (STRICT)
Use disciplined calibration to avoid inflated confidence:
- Start from 0.50 baseline.
- Add +0.05 to +0.10 for each clearly guideline-supported advantage.
- Subtract -0.05 to -0.15 for ambiguity, weak evidence, or unresolved contradictions.
- Return confidence rounded to 2 decimals.
- Do NOT output 1.00 unless essentially no uncertainty remains.
- Do NOT repeatedly use 0.75 as a default anchor; only use 0.75 if the evidence profile truly matches that certainty.

Confidence band mapping MUST be consistent:
- HIGH CONFIDENCE: confidence > 0.90
- MEDIUM CONFIDENCE: 0.60 <= confidence <= 0.90
- LOW CONFIDENCE: confidence < 0.60
"""


def create_judge_prompt_template_a1_first() -> str:
    """Primary judge prompt where A1 is shown before A2."""
    template = f"""# ROLE
You are **{{judge_name}}** - a Senior ACSA Annotation Judge, independent and impartial.

Task: Read both annotators' full label sets and the complete debate history, then select
**ONE annotator** whose OVERALL label set is more accurate.

---

# INPUT DATA

## 1. Review Text:
**"{{review_text}}"**

## 2. A1's Full Label Set (shown first in this view):
{{a1_labels_text}}

## 3. A2's Full Label Set (shown second in this view):
{{a2_labels_text}}

## 4. Relevant ACSA Guidelines (Retrieved from Knowledge Base):
{{retrieved_context}}

## 5. Debate History - Case 1 (A1 argues first):
{{case_1_history}}

## 6. Debate History - Case 2 (A2 argues first):
{{case_2_history}}

---

# TASK
1. Read the review text and both full label sets carefully.
2. Cross-check EACH label against the ACSA Guidelines.
3. Analyze which side's claims are more guideline-consistent.
4. Choose **ONE** winner: A1 or A2.
5. Provide a calibrated confidence score in [0, 1].

---

# PROCEDURE
Step 1: Identify differing labels between A1 and A2.

Step 2: For each differing label, check:
- Is A1 consistent with the guideline?
- Is A2 consistent with the guideline?
- Which side cited stronger direct guideline evidence?

Step 3: Overall decision:
- Pick the side that is correct on more differing labels with stronger evidence.

---

{_shared_confidence_rules()}

---

# CRITICAL RULES
- Only choose "A1" or "A2" as winner_annotator.
- Base decision on guideline compliance, not style or writing quality.
- reasoning and key_evidence MUST be in ENGLISH.

# OUTPUT FORMAT
{{{{
  "decision": {{{{
    "winner_annotator": "A1 or A2",
    "confidence_level": "HIGH | MEDIUM | LOW",
    "confidence": 0.00,
    "reasoning": "Why this annotator is more correct overall. Address key differing labels.",
    "key_evidence": "Most important direct guideline citation supporting the winner."
  }}}}
}}}}
"""
    return template


def create_judge_prompt_template_a2_first() -> str:
    """Primary judge prompt where A2 is shown before A1."""
    template = f"""# ROLE
You are **{{judge_name}}** - a Senior ACSA Annotation Judge, independent and impartial.

Task: Read both annotators' full label sets and the complete debate history, then select
**ONE annotator** whose OVERALL label set is more accurate.

---

# INPUT DATA

## 1. Review Text:
**"{{review_text}}"**

## 2. A2's Full Label Set (shown first in this view):
{{a2_labels_text}}

## 3. A1's Full Label Set (shown second in this view):
{{a1_labels_text}}

## 4. Relevant ACSA Guidelines (Retrieved from Knowledge Base):
{{retrieved_context}}

## 5. Debate History - Case 2 (A2 argues first):
{{case_2_history}}

## 6. Debate History - Case 1 (A1 argues first):
{{case_1_history}}

---

# TASK
1. Read the review text and both full label sets carefully.
2. Cross-check EACH label against the ACSA Guidelines.
3. Analyze which side's claims are more guideline-consistent.
4. Choose **ONE** winner: A1 or A2.
5. Provide a calibrated confidence score in [0, 1].

---

# PROCEDURE
Step 1: Identify differing labels between A1 and A2.

Step 2: For each differing label, check:
- Is A1 consistent with the guideline?
- Is A2 consistent with the guideline?
- Which side cited stronger direct guideline evidence?

Step 3: Overall decision:
- Pick the side that is correct on more differing labels with stronger evidence.

---

{_shared_confidence_rules()}

---

# CRITICAL RULES
- Only choose "A1" or "A2" as winner_annotator.
- Base decision on guideline compliance, not style or writing quality.
- reasoning and key_evidence MUST be in ENGLISH.

# OUTPUT FORMAT
{{{{
  "decision": {{{{
    "winner_annotator": "A1 or A2",
    "confidence_level": "HIGH | MEDIUM | LOW",
    "confidence": 0.00,
    "reasoning": "Why this annotator is more correct overall. Address key differing labels.",
    "key_evidence": "Most important direct guideline citation supporting the winner."
  }}}}
}}}}
"""
    return template


def create_judge_tiebreak_prompt_template() -> str:
    """Judge-3 tie-break prompt used only when two primary judges disagree with equal confidence."""
    template = """# ROLE
You are **{judge_name}** - a Final Tie-Break Judge for ACSA.

Two primary judges already reviewed the same case. They disagreed and reported equal confidence.
Your task is to make the final decision by auditing both judges' reasoning and key evidence.

---

# INPUT DATA

## 1. Review Text:
**"{review_text}"**

## 2. Relevant ACSA Guidelines (Retrieved Context):
{retrieved_context}

## 3. {primary_section_title_1}
- Winner: {primary_1_winner}
- Confidence: {primary_1_confidence}
- Reasoning:
{primary_1_reasoning}
- Key Evidence:
{primary_1_key_evidence}

## 4. {primary_section_title_2}
- Winner: {primary_2_winner}
- Confidence: {primary_2_confidence}
- Reasoning:
{primary_2_reasoning}
- Key Evidence:
{primary_2_key_evidence}

---

# TASK
1. Verify which judge's reasoning is more faithful to explicit guideline rules.
2. Resolve conflicts by prioritizing direct, verifiable guideline citations over vague arguments.
3. Output one final winner: A1 or A2.

---

# CRITICAL RULES
- You MUST choose exactly one winner: A1 or A2.
- Do NOT abstain.
- Do NOT invent labels.
- reasoning and key_evidence MUST be in ENGLISH.

# OUTPUT FORMAT
{{{{
  "decision": {{{{
    "winner_annotator": "A1 or A2",
    "reasoning": "Why this side should win after auditing both judges' arguments.",
    "key_evidence": "Most decisive guideline rule or citation that resolves the tie."
  }}}}
}}}}
"""
    return template
