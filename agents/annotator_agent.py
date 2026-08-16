import os
import sys
import json
import time
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ThreadTimeoutError
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from config.embedding_config import get_embeddings

from utils.helpers import load_prompt_from_yaml
from core_engine.conflict_filter import filter_and_route_conflict
from utils.token_usage_logger import log_llm_usage

load_dotenv(os.path.join(ROOT_DIR, ".env"))

ANNOTATOR_MAX_RETRIES = int(os.getenv("ANNOTATOR_MAX_RETRIES", "10"))
ANNOTATOR_BASE_SLEEP = float(os.getenv("ANNOTATOR_BASE_SLEEP", "1.0"))
ANNOTATOR_INTER_CALL_SLEEP = float(os.getenv("ANNOTATOR_INTER_CALL_SLEEP", "1"))
ANNOTATOR_INVOKE_TIMEOUT_SEC = float(os.getenv("ANNOTATOR_INVOKE_TIMEOUT_SEC", "60.0"))
ANNOTATOR_HEARTBEAT_INTERVAL_SEC = float(os.getenv("ANNOTATOR_HEARTBEAT_INTERVAL_SEC", "10.0"))


def _log_retry_failure(annotator_name: str, last_error: str, payload: dict):
    try:
        log_dir = os.path.join(ROOT_DIR, "system_data")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "annotator_retry_failures.jsonl")

        batch_text = payload.get("target_reviews_batch", "")
        review_ids = re.findall(r"Review ID:\s*(.+)", batch_text)

        entry = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "annotator": annotator_name,
            "max_retries": ANNOTATOR_MAX_RETRIES,
            "timeout_sec": ANNOTATOR_INVOKE_TIMEOUT_SEC,
            "last_error": last_error,
            "review_ids": review_ids[:50],
            "batch_chars": len(batch_text),
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Avoid crashing the pipeline if logging fails
        pass


def _build_annotator_input_text(payload: dict) -> str:
    return "\n\n".join([
        str(payload.get("target_reviews_batch", "")),
        str(payload.get("retrieved_guidelines", "")),
        str(payload.get("agreed_examples", "")),
    ])


def _normalize_label_item(item):
    if not isinstance(item, dict):
        return None

    entity = str(item.get("entity", "")).strip().upper()
    attribute = str(item.get("attribute", "")).strip().upper()
    sentiment = str(item.get("sentiment", "")).strip().upper()

    if not entity or not attribute or sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        return None

    return {
        "entity": entity,
        "attribute": attribute,
        "sentiment": sentiment,
    }


def _dedup_labels(labels):
    if not isinstance(labels, list):
        return []

    deduped = []
    seen = set()
    for raw in labels:
        norm = _normalize_label_item(raw)
        if not norm:
            continue

        key = (norm["entity"], norm["attribute"], norm["sentiment"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(norm)

    return deduped


def _sanitize_annotation_item(item):
    if not isinstance(item, dict):
        return None

    labels = _dedup_labels(item.get("labels", []))
    return {
        "review_id": item.get("review_id", ""),
        "review_text": item.get("review_text", ""),
        "labels": labels,
        "opinion": item.get("opinion", ""),
        "evidence": item.get("evidence", ""),
    }


def _sanitize_annotation_list(items):
    if not isinstance(items, list):
        return []

    sanitized = []
    for item in items:
        fixed = _sanitize_annotation_item(item)
        if fixed is None:
            continue
        sanitized.append(fixed)
    return sanitized


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
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return _sanitize_annotation_list([parsed])
        return _sanitize_annotation_list(parsed)
    except json.JSONDecodeError:
        return []


def _invoke_annotator_with_retry(chain, payload: dict, annotator_name: str):
    last_error = "Unknown error"

    for attempt in range(1, ANNOTATOR_MAX_RETRIES + 1):
        try:
            response = chain.invoke(payload)
            input_text = _build_annotator_input_text(payload)
            output_text = getattr(response, "content", "") if response is not None else ""
            log_llm_usage(annotator_name, input_text, str(output_text), response_obj=response, call_type=annotator_name)
            parsed = clean_json_output(response.content)
            if isinstance(parsed, list):
                return parsed

            last_error = "Output is not a valid list JSON"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[{annotator_name}] Exception detail: {last_error}", flush=True)

        if attempt < ANNOTATOR_MAX_RETRIES:
            wait_s = ANNOTATOR_BASE_SLEEP * attempt
            print(
                f"[{annotator_name}] Retry {attempt}/{ANNOTATOR_MAX_RETRIES} sau loi: {last_error}. "
                f"Sleep {wait_s:.1f}s...",
                flush=True
            )
            time.sleep(wait_s)

    print(f"[{annotator_name}] That bai sau {ANNOTATOR_MAX_RETRIES} lan thu. Loi cuoi: {last_error}", flush=True)
    _log_retry_failure(annotator_name, last_error, payload)
    return []


def _invoke_annotator_with_timeout_and_retry(chain, payload: dict, annotator_name: str):
    """Wrap invoke with timeout to prevent terminal freeze on network lag."""
    last_error = "Unknown error"

    for attempt in range(1, ANNOTATOR_MAX_RETRIES + 1):
        try:
            print(f"[{annotator_name}] Attempt {attempt}/{ANNOTATOR_MAX_RETRIES} (timeout={ANNOTATOR_INVOKE_TIMEOUT_SEC}s)...", flush=True)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(chain.invoke, payload)
                response = future.result(timeout=ANNOTATOR_INVOKE_TIMEOUT_SEC)

            input_text = _build_annotator_input_text(payload)
            output_text = getattr(response, "content", "") if response is not None else ""
            log_llm_usage(annotator_name, input_text, str(output_text), response_obj=response, call_type=annotator_name)

            parsed = clean_json_output(response.content)
            if isinstance(parsed, list):
                print(f"[{annotator_name}] Success on attempt {attempt}.", flush=True)
                return parsed

            last_error = "Output is not a valid list JSON"

        except ThreadTimeoutError:
            last_error = f"Timeout after {ANNOTATOR_INVOKE_TIMEOUT_SEC}s (network lag or API slow)"
            print(f"[{annotator_name}] {last_error}", flush=True)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[{annotator_name}] Exception detail: {last_error}", flush=True)

        if attempt < ANNOTATOR_MAX_RETRIES:
            wait_s = ANNOTATOR_BASE_SLEEP * attempt
            print(
                f"[{annotator_name}] Retry {attempt}/{ANNOTATOR_MAX_RETRIES} sau loi: {last_error}. "
                f"Sleep {wait_s:.1f}s...",
                flush=True
            )
            time.sleep(wait_s)

    print(f"[{annotator_name}] That bai sau {ANNOTATOR_MAX_RETRIES} lan thu. Loi cuoi: {last_error}", flush=True)
    _log_retry_failure(annotator_name, last_error, payload)
    return []


def get_retrieved_context_for_batch(batch_data: list, base_db_dir: str):
    embeddings = get_embeddings()
    combined_review_text = " ".join([item["text"] for item in batch_data])

    guideline_db_path = os.path.join(base_db_dir, "chroma_db")
    guidelines_text = "No guideline available."

    if os.path.exists(guideline_db_path) and len(os.listdir(guideline_db_path)) > 0:
        try:
            vector_db = Chroma(persist_directory=guideline_db_path, embedding_function=embeddings)
            docs = vector_db.similarity_search(combined_review_text, k=4)
            if docs:
                guidelines_text = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi truy xuất Guideline DB: {e}")
    else:
        print("[THÔNG BÁO] Không tìm thấy Knowledge DB (chroma_db).")

    agreed_db_path = os.path.join(base_db_dir, "chroma_db_agreed")
    agreed_examples_text = "No agreed examples available for this context."

    if os.path.exists(agreed_db_path) and len(os.listdir(agreed_db_path)) > 0:
        try:
            agreed_vector_db = Chroma(persist_directory=agreed_db_path, embedding_function=embeddings)
            agreed_docs = agreed_vector_db.similarity_search(combined_review_text, k=2)
            if agreed_docs:
                agreed_examples_text = "\n\n".join([doc.page_content for doc in agreed_docs])
                print("[RAG System] Retrieved agreed-case examples successfully.")
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi truy xuất agreed-case DB: {e}")
    else:
        print("[RAG System] No agreed-case DB found. The system will only retrieve guideline context.")

    return guidelines_text, agreed_examples_text


def annotate_with_agent1(batch_text_prompt: str, retrieved_guidelines: str, agreed_examples: str):
    deployment_name = os.getenv("ANNOTATOR_1_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    temperature = float(os.getenv("ANNOTATOR_1_TEMPERATURE", "0.1"))
    llm = AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("BASE_URL") or os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("API_VERSION"),
        azure_deployment=deployment_name,
        temperature=temperature,
    )

    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)

    chain = prompt | llm
    return _invoke_annotator_with_retry(
        chain,
        {
            "target_reviews_batch": batch_text_prompt,
            "retrieved_guidelines": retrieved_guidelines,
            "agreed_examples": agreed_examples,
        },
        annotator_name="Annotator 1",
    )


def annotate_with_agent1_timeout(batch_text_prompt: str, retrieved_guidelines: str, agreed_examples: str):
    deployment_name = os.getenv("ANNOTATOR_1_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    temperature = float(os.getenv("ANNOTATOR_1_TEMPERATURE", "0.1"))
    llm = AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("BASE_URL") or os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("API_VERSION"),
        azure_deployment=deployment_name,
        temperature=temperature,
        request_timeout=ANNOTATOR_INVOKE_TIMEOUT_SEC,
    )

    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)

    chain = prompt | llm
    return _invoke_annotator_with_timeout_and_retry(
        chain,
        {
            "target_reviews_batch": batch_text_prompt,
            "retrieved_guidelines": retrieved_guidelines,
            "agreed_examples": agreed_examples,
        },
        annotator_name="Annotator 1",
    )


def annotate_with_agent2(batch_text_prompt: str, retrieved_guidelines: str, agreed_examples: str):
    deployment_name = os.getenv("ANNOTATOR_2_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
    temperature = float(os.getenv("ANNOTATOR_2_TEMPERATURE", "0.4"))
    llm = AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("BASE_URL") or os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("API_VERSION"),
        azure_deployment=deployment_name,
        temperature=temperature,
    )

    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)

    chain = prompt | llm
    return _invoke_annotator_with_retry(
        chain,
        {
            "target_reviews_batch": batch_text_prompt,
            "retrieved_guidelines": retrieved_guidelines,
            "agreed_examples": agreed_examples,
        },
        annotator_name="Annotator 2",
    )


def annotate_with_agent2_timeout(batch_text_prompt: str, retrieved_guidelines: str, agreed_examples: str):
    deployment_name = os.getenv("ANNOTATOR_2_DEPLOYMENT") or os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
    temperature = float(os.getenv("ANNOTATOR_2_TEMPERATURE", "0.4"))
    llm = AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("BASE_URL") or os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("API_VERSION"),
        azure_deployment=deployment_name,
        temperature=temperature,
        request_timeout=ANNOTATOR_INVOKE_TIMEOUT_SEC,
    )

    yaml_path = os.path.join(ROOT_DIR, "prompts", "agent_prompt.yaml")
    prompt_str = load_prompt_from_yaml(yaml_path, "annotator_agent", "system_prompt")
    prompt = ChatPromptTemplate.from_template(prompt_str)

    chain = prompt | llm
    return _invoke_annotator_with_timeout_and_retry(
        chain,
        {
            "target_reviews_batch": batch_text_prompt,
            "retrieved_guidelines": retrieved_guidelines,
            "agreed_examples": agreed_examples,
        },
        annotator_name="Annotator 2",
    )


def process_and_verify_batch(batch_data: list, base_db_dir: str):
    api_key_openai = os.getenv("OPENAI_API_KEY")

    if not api_key_openai:
        raise ValueError("[LOI] Ban can khai bao OPENAI_API_KEY trong file .env")

    batch_text_prompt = ""
    for item in batch_data:
        batch_text_prompt += f"Review ID: {item['id']}\nText: {item['text']}\n---\n"

    print(f"\n[RAG System] Đang phân tích Batch ({len(batch_data)} câu)...", flush=True)
    rules, agreed_ex = get_retrieved_context_for_batch(batch_data, base_db_dir)

    print(f"[Annotator 1] Azure Model (Temp=0.1) đang gán nhãn cho {len(batch_data)} câu...", flush=True)
    a1_batch_result = annotate_with_agent1_timeout(batch_text_prompt, rules, agreed_ex)

    if ANNOTATOR_INTER_CALL_SLEEP > 0:
        print(f"[Annotator] Sleep {ANNOTATOR_INTER_CALL_SLEEP:.1f}s trước khi gọi Annotator 2...", flush=True)
        time.sleep(ANNOTATOR_INTER_CALL_SLEEP)

    print(f"[Annotator 2] Azure Model (Temp=0.4) đang gán nhãn cho {len(batch_data)} câu...", flush=True)
    a2_batch_result = annotate_with_agent2_timeout(batch_text_prompt, rules, agreed_ex)

    if not isinstance(a1_batch_result, list):
        a1_batch_result = []
    if not isinstance(a2_batch_result, list):
        a2_batch_result = []

    final_batch_results = []

    for item in batch_data:
        rev_id = item["id"]
        rev_text = item["text"]

        a1_data_for_id = next((x for x in a1_batch_result if x.get("review_id") == rev_id), {})
        a2_data_for_id = next((x for x in a2_batch_result if x.get("review_id") == rev_id), {})

        print(f"\n[Kiểm tra] So sánh kết quả cho {rev_id}", flush=True)
        res = filter_and_route_conflict(rev_id, rev_text, a1_data_for_id, a2_data_for_id)
        final_batch_results.append(res)

    return final_batch_results


if __name__ == "__main__":
    base_system_dir = os.path.join(ROOT_DIR, "system_data")

    test_batch = [
        {"id": "REV_001", "text": "Phòng ốc rộng rãi, sạch sẽ nhưng thái độ nhân viên lễ tân hơi kém."},
        {"id": "REV_002", "text": "Đồ ăn sáng cực kỳ ngon miệng, buffet đa dạng. Cảnh biển đẹp tuyệt vời."},
        {"id": "REV_003", "text": "Wifi khách sạn rất chậm, mình không thể làm việc được."},
    ]

    results = process_and_verify_batch(batch_data=test_batch, base_db_dir=base_system_dir)
    print("\n[KET QUA TRA VE CHO HE THONG]")
    print(json.dumps(results, indent=2, ensure_ascii=False))
