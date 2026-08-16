import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.helpers import load_prompt_from_yaml
from utils.llm_invoke_with_timeout import invoke_with_timeout_and_retry

load_dotenv(os.path.join(ROOT_DIR, ".env"))


def clean_json_output(output: str):
    output = output.strip()
    if output.startswith("```json"):
        output = output.replace("```json\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]
    elif output.startswith("```"):
        output = output.replace("```\n", "", 1)
        if output.endswith("```"):
            output = output[:-3]
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw_output": output}


def propose_consolidated_guideline_update(
    root_cause_records: list,
    current_guideline: str,
    target_domain: str,
):
    """Create one cycle-level suggestion report from all conflict root causes."""
    if not root_cause_records:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("BASE_URL")
    if not api_key or not endpoint:
        raise ValueError("[ERROR] Missing OPENAI_API_KEY or BASE_URL in .env")

    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=os.getenv("API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini"),
        temperature=0.3,
    )

    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(
        yaml_path,
        "guideline_agent_consolidated",
        "system_prompt",
    )
    prompt = ChatPromptTemplate.from_template(prompt_str)
    payload = {
        "target_domain": target_domain,
        "current_guideline": current_guideline,
        "root_cause_count": len(root_cause_records),
        "root_cause_records": json.dumps(
            root_cause_records,
            ensure_ascii=False,
            indent=2,
        ),
    }

    print(
        "\n[Guideline Agent] Creating one consolidated suggestion report "
        f"from {len(root_cause_records)} root-cause records..."
    )
    messages = prompt.format_messages(**payload)
    response = invoke_with_timeout_and_retry(
        llm,
        messages,
        agent_name="Guideline Agent",
        call_type="guideline_agent_consolidated",
    )
    output_text = response.content if (response is not None and hasattr(response, "content")) else ""
    return clean_json_output(output_text)


def append_to_guideline_file(proposal: dict, filepath: str):
    """Append the human-approved consolidated update to the active guideline."""
    if not proposal:
        return False

    rule_content = proposal.get("candidate_guideline_text", "")
    if not isinstance(rule_content, str) or not rule_content.strip():
        return False

    location = proposal.get("location_in_guideline", "HUMAN-APPROVED CYCLE UPDATE")
    action = proposal.get("action_type", "CONSOLIDATED UPDATE")
    formatted_rule = f"\n\n### {action} TARGET: {location} ###\n{rule_content.strip()}"

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(formatted_rule)
        print(f"[GUIDELINE] Applied approved update to: {os.path.basename(filepath)}")
        return True
    except OSError as exc:
        print(f"[ERROR] Cannot update guideline file: {exc}")
        return False
