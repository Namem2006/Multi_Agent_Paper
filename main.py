import os
import sys
import json
import math
import time
import traceback
import warnings

# Tat canh bao Pydantic
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from agents.adapt_agent import generate_adapted_guideline
from rag_system.build_knowledge_base import build_vector_database
from agents.annotator_agent import process_and_verify_batch
from rag_system.build_agreed_db import build_agreed_db_from_scratch
from core_engine.data_loader import extract_and_assign_ids
from core_engine.workflow_controller import run_full_conflict_workflow
from core_engine.update_guideline import process_all_causes
from config.embedding_config import get_embeddings  # Import để pre-load
from utils.token_usage_logger import reset_token_usage
from utils.foody_preprocessor import ensure_foody_dataset

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"last_processed_index": 0, "active_guideline_path": "", "active_domain": ""}

def save_progress(progress_file, progress_data):
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=4, ensure_ascii=False)

DEFAULT_RUN_SIZE = 30
ANNOTATION_CHUNK_SIZE = 3
DEFAULT_SOURCE_DOMAIN = "Restaurant"
DEFAULT_TARGET_DOMAIN = "Hotel"


def read_positive_int(prompt: str, default_value: int, max_value: int | None = None) -> int:
    raw_value = input(prompt).strip()
    if not raw_value:
        return default_value

    try:
        value = int(raw_value)
    except ValueError:
        print(f"[WARN] Invalid number. Using default: {default_value}.")
        return default_value

    if value <= 0:
        print(f"[WARN] Value must be positive. Using default: {default_value}.")
        return default_value

    if max_value is not None and value > max_value:
        print(f"[WARN] Value exceeds remaining samples. Using: {max_value}.")
        return max_value

    return value


def choose_guideline_source(all_sample_reviews, source_guideline_path: str, adapted_guideline_path: str):
    print("\n[GUIDELINE] Choose the active guideline source:")
    print("[1] Use existing guideline file (data/guideline.txt)")
    print("[2] Generate adapted guideline with Adapt Agent")

    choice = input("Enter choice (1 or 2, default=1): ").strip()
    if choice == "2":
        custom_source_path = input(
            f"Enter source guideline path (default={source_guideline_path}): "
        ).strip()
        if custom_source_path:
            source_guideline_path = custom_source_path
        if not os.path.exists(source_guideline_path):
            raise FileNotFoundError(f"Source guideline not found: {source_guideline_path}")

        source_domain = input(
            f"Enter source guideline domain (default={DEFAULT_SOURCE_DOMAIN}): "
        ).strip() or DEFAULT_SOURCE_DOMAIN
        target_domain = input(
            f"Enter target domain to adapt into (default={DEFAULT_TARGET_DOMAIN}): "
        ).strip() or DEFAULT_TARGET_DOMAIN

        print("\n[STEP 1] Running Adapt Agent...")
        sample_count = min(DEFAULT_RUN_SIZE, len(all_sample_reviews))
        samples_str = "\n".join(
            f"{idx + 1}. {item['text']}"
            for idx, item in enumerate(all_sample_reviews[:sample_count])
        )
        generate_adapted_guideline(
            source_file_path=source_guideline_path,
            source_domain=source_domain,
            target_domain=target_domain,
            samples=samples_str,
            output_file_path=adapted_guideline_path,
        )
        active_guideline_path = adapted_guideline_path
    else:
        target_domain = input(
            f"Enter domain name for the existing guideline (default={DEFAULT_SOURCE_DOMAIN}): "
        ).strip() or DEFAULT_SOURCE_DOMAIN
        active_guideline_path = source_guideline_path

    if not os.path.exists(active_guideline_path):
        raise FileNotFoundError(f"Active guideline not found: {active_guideline_path}")

    print(f"[GUIDELINE] Active guideline: {active_guideline_path}")
    return active_guideline_path, target_domain


def choose_run_mode(start_idx: int, total_dataset_size: int):
    remaining = total_dataset_size - start_idx
    default_count = min(DEFAULT_RUN_SIZE, remaining)

    print("\n[RUN MODE] Choose how this pipeline run should execute:")
    print("[1] Annotation-only: run a chosen number of samples; skip Root-cause Agent and guideline update")
    print("[2] Guideline-update mode: run one fixed N-sample cycle; run Root-cause Agent and one human guideline review")

    mode_choice = input("Enter choice (1 or 2, default=1): ").strip()
    enable_guideline_update = mode_choice == "2"

    if enable_guideline_update:
        run_count = read_positive_int(
            f"Enter N samples for one update cycle (default={default_count}): ",
            default_value=default_count,
            max_value=remaining,
        )
        mode_name = "guideline-update"
    else:
        run_count = read_positive_int(
            f"Enter number of samples to annotate (default={default_count}): ",
            default_value=default_count,
            max_value=remaining,
        )
        mode_name = "annotation-only"

    run_start = start_idx
    run_end = min(start_idx + run_count, total_dataset_size)
    return mode_name, enable_guideline_update, run_start, run_end


def reset_current_run_logs(conflict_log_path: str, cause_data_path: str):
    if os.path.exists(conflict_log_path):
        os.remove(conflict_log_path)
    if os.path.exists(cause_data_path):
        os.remove(cause_data_path)


def load_conflict_samples(conflict_log_path: str):
    conflict_samples = []
    if not os.path.exists(conflict_log_path):
        return conflict_samples

    with open(conflict_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conflict_samples.append(json.loads(line))
    return conflict_samples


def prepare_conflict_for_debate(conflict_data: dict, fallback_index: int):
    sid = conflict_data.get("review_id", f"conflict_{fallback_index}")
    conflict_data["sample_id"] = sid
    conflict_data["A1_initial"] = {
        "labels": conflict_data.get("a1_labels", []),
        "opinion": conflict_data.get("a1_opinion") or "I selected this label based on the review text.",
        "evidence": conflict_data.get("a1_evidence") or "According to the active guideline.",
    }
    conflict_data["A2_initial"] = {
        "labels": conflict_data.get("a2_labels", []),
        "opinion": conflict_data.get("a2_opinion") or "I selected a different interpretation.",
        "evidence": conflict_data.get("a2_evidence") or "According to the active guideline.",
    }
    return conflict_data


def run_annotation_stage_v2(run_samples, run_start: int, system_data_dir: str):
    total_turns = math.ceil(len(run_samples) / ANNOTATION_CHUNK_SIZE)
    global_agreed = 0
    global_conflict = 0

    print(
        f"\n[STEP 3] Start annotation "
        f"({total_turns} turns, {ANNOTATION_CHUNK_SIZE} samples/turn)"
    )

    for turn in range(total_turns):
        turn_start = turn * ANNOTATION_CHUNK_SIZE
        turn_end = min(turn_start + ANNOTATION_CHUNK_SIZE, len(run_samples))
        current_batch = run_samples[turn_start:turn_end]

        print("\n" + "-" * 50)
        print(
            f"Turn {turn + 1}/{total_turns} | "
            f"Dataset rows {run_start + turn_start + 1} to {run_start + turn_end}"
        )
        print("-" * 50)

        for item in current_batch:
            short_text = item["text"][:60] + "..." if len(item["text"]) > 60 else item["text"]
            print(f"--- {item['id']}: {short_text}")

        results = process_and_verify_batch(batch_data=current_batch, base_db_dir=system_data_dir)

        turn_agreed = sum(1 for r in results if r["status"] == "AGREED")
        turn_conflict = len(results) - turn_agreed
        global_agreed += turn_agreed
        global_conflict += turn_conflict

        if turn_agreed > 0:
            print(f"-> {turn_agreed} agreed samples. Rebuilding agreed-case DB...")
            build_agreed_db_from_scratch()

    return global_agreed, global_conflict


def run_conflict_stage_v2(conflict_log_path: str, enable_guideline_update: bool):
    conflict_samples = load_conflict_samples(conflict_log_path)
    if not conflict_samples:
        print("\n[STEP 4] No conflict samples to resolve.")
        return 0

    print(f"\n[STEP 4] Running debate/judge workflow for {len(conflict_samples)} conflicts...")

    for idx, conflict_data in enumerate(conflict_samples, start=1):
        conflict_data = prepare_conflict_for_debate(conflict_data, idx)
        sid = conflict_data.get("sample_id", f"conflict_{idx}")
        review_text = conflict_data.get("review", "")

        print(f"\n  [{idx}/{len(conflict_samples)}] {sid}: {review_text[:60]!r} ...")

        try:
            workflow_output = run_full_conflict_workflow(
                conflict_data=conflict_data,
                max_rounds=2,
                enable_root_cause=enable_guideline_update,
            )
            if workflow_output:
                final_decision = workflow_output["judge_result"]["final_decision"]
                print(f"  -> Judge winner: {final_decision['winner_annotator']}")
        except Exception as exc:
            print(f"  -> ERROR in workflow for {sid}: {exc}")

        time.sleep(1.0)

    return len(conflict_samples)

def run_workflow():
    return run_workflow_v2()

    data_dir = os.path.join(ROOT_DIR, "data")
    system_data_dir = os.path.join(ROOT_DIR, "system_data")
    reset_token_usage(system_data_dir)
    output_dir = os.path.join(system_data_dir, "result")

    os.makedirs(system_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raw_dataset_path = os.path.join(data_dir, "Data_Foody_Final.txt")
    dataset_path = os.path.join(data_dir, "Data_Foody_Final_with_id.txt")
    source_guideline_path = os.path.join(data_dir, "guideline.txt")
    adapted_guideline_path = os.path.join(system_data_dir, "adapted_guideline.txt")

    db_directory = os.path.join(system_data_dir, "chroma_db")
    conflict_log_path = os.path.join(system_data_dir, "conflict_samples.jsonl")
    cause_data_path = os.path.join(system_data_dir, "cause", "cause_data.json")
    progress_file_path = os.path.join(system_data_dir, "progress.json")

    # LUÔN XÓA LOG XUNG ĐỘT CŨ Ở MỖI LẦN CHẠY N MẪU (để chỉ giải quyết xung đột của lượt này)
    if os.path.exists(conflict_log_path):
        os.remove(conflict_log_path)
    if os.path.exists(cause_data_path):
        os.remove(cause_data_path)

    try:
        ensure_foody_dataset(raw_dataset_path, dataset_path)
    except Exception as exc:
        print(f"[LỖI] Không thể chuẩn hoá Foody dataset: {exc}")
        return

    # 1. LOAD TIẾN ĐỘ VÀ DỮ LIỆU
    progress = load_progress(progress_file_path)
    start_idx = progress["last_processed_index"]

    print("\n" + "="*60)
    print(" HỆ THỐNG GÁN NHÃN ĐA TÁC VỤ (MULTI-AGENT ACSA) ")
    print("="*60)

    # PRE-LOAD EMBEDDING MODEL NGAY TỪ ĐẦU (để tránh lỗi khi chạy workflow)
    print("\n[KHỞI TẠO] Đang pre-load embedding model...")
    try:
        get_embeddings()  # Load model 1 lần duy nhất
        print("[KHỞI TẠO] ✓ Model đã sẵn sàng.\n")
    except Exception as e:
        print(f"[LỖI] Không thể load embedding model: {e}")
        print("Vui lòng kiểm tra kết nối mạng hoặc cache HuggingFace.")
        return

    print(f"[DATA] Đang tải dataset từ {os.path.basename(dataset_path)} ...")
    all_sample_reviews = extract_and_assign_ids(dataset_path)
    total_dataset_size = len(all_sample_reviews)

    if total_dataset_size == 0:
        print("[LỖI] Không thể đọc được dữ liệu. Kiểm tra lại file txt.")
        return

    if start_idx >= total_dataset_size:
        print(f"\n[HOÀN TẤT] Bạn đã chạy xong toàn bộ {total_dataset_size} câu trong Dataset!")
        print("Nếu muốn chạy lại từ đầu, hãy xóa file 'system_data/progress.json'.")
        return

    print(f"[TIẾN ĐỘ] Hệ thống đang ở mốc: Đã xử lý {start_idx} / {total_dataset_size} câu.")

    # 2. HỎI RANGE MẪU MUỐN CHẠY (TỪ - ĐẾN)
    print(f"\n[CHỌN RANGE] Nhập range mẫu muốn chạy trong lượt này:")
    print(f"   Ví dụ: '1 30' để chạy từ câu 1 đến 30")
    print(f"   Hoặc '31 60' để chạy từ câu 31 đến 60")
    print(f"   Mặc định: Từ {start_idx + 1} đến {start_idx + 30}")

    try:
        range_str = input("Nhập range (từ đến) hoặc bấm Enter để mặc định 30 câu: ").strip()

        if range_str:
            parts = range_str.split()
            if len(parts) == 2:
                from_idx = int(parts[0]) - 1  # Convert 1-indexed to 0-indexed
                to_idx = int(parts[1])

                # Validate range
                if from_idx < 0:
                    print("[CẢNH BÁO] Câu bắt đầu phải >= 1. Điều chỉnh về 1.")
                    from_idx = 0
                if to_idx > total_dataset_size:
                    print(f"[CẢNH BÁO] Câu kết thúc vượt quá {total_dataset_size}. Điều chỉnh về {total_dataset_size}.")
                    to_idx = total_dataset_size
                if from_idx >= to_idx:
                    print("[LỖI] Câu bắt đầu phải nhỏ hơn câu kết thúc. Dùng mặc định 30 câu.")
                    from_idx = start_idx
                    to_idx = min(start_idx + 30, total_dataset_size)

                run_start = from_idx
                run_end = to_idx
            else:
                print("[LỖI] Vui lòng nhập đúng format 'từ đến'. Dùng mặc định 30 câu.")
                run_start = start_idx
                run_end = min(start_idx + 30, total_dataset_size)
        else:
            # Default: 30 samples
            run_start = start_idx
            run_end = min(start_idx + 30, total_dataset_size)
    except ValueError:
        print("[LỖI] Input không hợp lệ. Dùng mặc định 30 câu.")
        run_start = start_idx
        run_end = min(start_idx + 30, total_dataset_size)

    run_samples = all_sample_reviews[run_start:run_end]
    actual_run_count = len(run_samples)

    print(f"\n-> Phiên này sẽ chạy {actual_run_count} câu (Từ câu {run_start + 1} đến {run_end}).")

    # 3. QUẢN LÝ GUIDELINE
    active_guideline_path = progress.get("active_guideline_path")

    if start_idx == 0 or not active_guideline_path or not os.path.exists(active_guideline_path):
        # Chỉ hỏi Adapt Domain nếu là lần chạy đầu tiên
        print("\n[CẤU HÌNH LẦN ĐẦU] Chọn chế độ khởi tạo:")
        print("[1]. Domain mới (Chạy Adapt Agent tự sinh luật từ data mẫu)")
        print("[2]. Domain đã có luật (Dùng file guideline.txt có sẵn)")
        choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()

        if choice == '1':
            print("\n[BUOC 1] ADAPT AGENT (DOMAIN ADAPTATION)...")
            samples_str = "\n".join([f"{idx+1}. {item['text']}" for idx, item in enumerate(all_sample_reviews[:30])])
            generate_adapted_guideline(
                source_file_path=source_guideline_path, target_domain="Restaurant",
                samples=samples_str, output_file_path=adapted_guideline_path
            )
            active_guideline_path = adapted_guideline_path
        else:
            active_guideline_path = source_guideline_path

        progress["active_guideline_path"] = active_guideline_path
        save_progress(progress_file_path, progress)
    else:
        print(f"\n[GUIDELINE] Đang sử dụng bộ luật hiện tại: {os.path.basename(active_guideline_path)}")

    # 4. NẠP LẠI VECTOR DB VỚI LUẬT MỚI NHẤT
    print("\n[BUOC 2] NẠP BỘ LUẬT VÀO VECTOR DB (RAG)...")
    build_vector_database(file_path=active_guideline_path, persist_directory=db_directory)

    # 5. CHẠY ANNOTATOR (Vẫn chia lô 3 câu/lần)
    chunk_size = 3
    total_turns = math.ceil(actual_run_count / chunk_size)

    global_agreed = 0
    global_conflict = 0

    print(f"\n[BUOC 3] BẮT ĐẦU GÁN NHÃN ({total_turns} Lượt, mỗi lượt {chunk_size} câu)")

    for turn in range(total_turns):
        turn_start = turn * chunk_size
        turn_end = min(turn_start + chunk_size, actual_run_count)
        current_batch = run_samples[turn_start:turn_end]

        print(f"\n{'-'*50}")
        print(f" LƯỢT {turn + 1}/{total_turns} | Đang xử lý câu {run_start + turn_start + 1} đến {run_start + turn_end} ")
        print(f"{'-'*50}")

        for item in current_batch:
            short_text = item['text'][:60] + "..." if len(item['text']) > 60 else item['text']
            print(f"--- {item['id']}: {short_text}")

        results = process_and_verify_batch(batch_data=current_batch, base_db_dir=system_data_dir)

        turn_agreed = sum(1 for r in results if r["status"] == "AGREED")
        turn_conflict = len(results) - turn_agreed
        global_agreed += turn_agreed
        global_conflict += turn_conflict

        if turn_agreed > 0:
            print(f"-> Đồng thuận {turn_agreed} câu. Nạp án lệ vào agreed-case DB...")
            build_agreed_db_from_scratch()

    print("\n" + "="*50)
    print(f" TỔNG KẾT GIAI ĐOẠN GÁN NHÃN (Lô {actual_run_count} câu) ")
    print(f" Đồng thuận: {global_agreed} | Xung đột: {global_conflict} ")
    print("="*50)

    # 6. LUỒNG TRANH BIỆN & CẬP NHẬT LUẬT CHO N MẪU NÀY
    if global_conflict > 0:
        print(f"\n[BUOC 5] CHẠY LUỒNG TRANH BIỆN CHO {global_conflict} CÂU XUNG ĐỘT...")

        conflict_samples = []
        if os.path.exists(conflict_log_path):
            with open(conflict_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): conflict_samples.append(json.loads(line))

        for j, cdata in enumerate(conflict_samples):
            sid = cdata.get("review_id", f"conflict_{j+1}")
            cdata["sample_id"] = sid
            cdata["A1_initial"] = {
                "labels": cdata.get("a1_labels", []),
                "opinion": cdata.get("a1_opinion") or "Tôi chọn nhãn này dựa trên text.",
                "evidence": cdata.get("a1_evidence") or "Theo quy định."
            }
            cdata["A2_initial"] = {
                "labels": cdata.get("a2_labels", []),
                "opinion": cdata.get("a2_opinion") or "Tôi có quan điểm khác.",
                "evidence": cdata.get("a2_evidence") or "Theo quy định."
            }

            review_text = cdata.get("review", "")
            print(f"\n  [{j+1}/{len(conflict_samples)}] {sid}: {review_text[:60]!r} ...")

            try:
                workflow_output = run_full_conflict_workflow(conflict_data=cdata, max_rounds=2)
                if workflow_output:
                    dr = workflow_output["judge_result"]
                    print(f"  -> Judge Winner: {dr['final_decision']['winner_annotator']}")
            except Exception as exc:
                print(f"  -> ERROR in workflow for {sid}: {exc}")
            time.sleep(1.0)

        print("\n[BUOC 6] GUIDELINE AGENT - ĐỀ XUẤT CẬP NHẬT LUẬT TỪ CÁC LỖI VỪA RỒI")
        process_all_causes(active_guideline_path=active_guideline_path, target_domain="Restaurant")

    else:
        print("\n[BUOC 5 & 6] Bỏ qua (Không có xung đột nào trong lô này).")

    # 7. LƯU LẠI TIẾN ĐỘ
    progress["last_processed_index"] = run_end
    save_progress(progress_file_path, progress)

    print("\n" + "*"*60)
    print(f" ĐÃ LƯU TIẾN ĐỘ: Hoàn thành đến câu {run_end}/{total_dataset_size}. ")
    print(" Luật mới (nếu có) đã được cập nhật sẵn sàng cho lượt chạy sau. ")
    print("*"*60)

    # 8. HỎI NGƯỜI DÙNG CÓ MUỐN CHẠY TIẾP KHÔNG MÀ KHÔNG CẦN TẮT SCRIPT
    if run_end < total_dataset_size:
        cont = input("\nBạn có muốn tự động chạy tiếp lô mẫu tiếp theo không? (y/n): ").strip().lower()
        if cont == 'y':
            print("\n" + "~"*60)
            run_workflow() # Gọi đệ quy để chạy tiếp
        else:
            print("Đã dừng hệ thống. Bạn có thể kiểm tra file luật và kết quả. Chạy lại script để tiếp tục.")

def select_active_guideline_for_run(
    progress: dict,
    all_sample_reviews: list,
    source_guideline_path: str,
    adapted_guideline_path: str,
    reuse_active_guideline: bool = False,
):
    active_guideline_path = progress.get("active_guideline_path", "")
    target_domain = progress.get("active_domain", "") or DEFAULT_SOURCE_DOMAIN

    if active_guideline_path and os.path.exists(active_guideline_path):
        if reuse_active_guideline:
            print("\n[GUIDELINE] Reusing active guideline from previous cycle.")
            print(f"Path: {active_guideline_path}")
            print(f"Domain: {target_domain}")
            return active_guideline_path, target_domain

        print("\n[GUIDELINE] Existing active guideline found in progress:")
        print(f"Path: {active_guideline_path}")
        print(f"Domain: {target_domain}")
        use_current = input("Use this active guideline? (Y/n): ").strip().lower()
        if use_current in ["", "y", "yes"]:
            return active_guideline_path, target_domain

    return choose_guideline_source(
        all_sample_reviews=all_sample_reviews,
        source_guideline_path=source_guideline_path,
        adapted_guideline_path=adapted_guideline_path,
    )


def run_workflow_v2(reuse_active_guideline: bool = False):
    data_dir = os.path.join(ROOT_DIR, "data")
    system_data_dir = os.path.join(ROOT_DIR, "system_data")
    output_dir = os.path.join(system_data_dir, "result")

    os.makedirs(system_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    reset_token_usage(system_data_dir)

    raw_dataset_path = os.path.join(data_dir, "Data_Foody_Final.txt")
    dataset_path = os.path.join(data_dir, "Data_Foody_Final_with_id.txt")
    source_guideline_path = os.path.join(data_dir, "guideline.txt")
    adapted_guideline_path = os.path.join(system_data_dir, "adapted_guideline.txt")
    db_directory = os.path.join(system_data_dir, "chroma_db")
    conflict_log_path = os.path.join(system_data_dir, "conflict_samples.jsonl")
    cause_data_path = os.path.join(system_data_dir, "cause", "cause_data.json")
    progress_file_path = os.path.join(system_data_dir, "progress.json")

    print("\n" + "=" * 60)
    print(" MULTI-AGENT ACSA ANNOTATION SYSTEM ")
    print("=" * 60)

    try:
        ensure_foody_dataset(raw_dataset_path, dataset_path)
    except Exception as exc:
        print(f"[ERROR] Cannot prepare Foody dataset: {exc}")
        return

    print("\n[INIT] Pre-loading embedding model...")
    try:
        get_embeddings()
        print("[INIT] Embedding model is ready.\n")
    except Exception as exc:
        print(f"[ERROR] Cannot load embedding model: {exc}")
        return

    print(f"[DATA] Loading dataset from {os.path.basename(dataset_path)} ...")
    all_sample_reviews = extract_and_assign_ids(dataset_path)
    total_dataset_size = len(all_sample_reviews)
    if total_dataset_size == 0:
        print("[ERROR] Dataset is empty or unreadable.")
        return

    progress = load_progress(progress_file_path)
    start_idx = int(progress.get("last_processed_index", 0))
    if start_idx >= total_dataset_size:
        print(f"\n[DONE] All {total_dataset_size} samples have been processed.")
        print("Delete system_data/progress.json if you want to start from the beginning.")
        return

    print(f"[PROGRESS] Processed {start_idx}/{total_dataset_size} samples.")

    active_guideline_path, target_domain = select_active_guideline_for_run(
        progress=progress,
        all_sample_reviews=all_sample_reviews,
        source_guideline_path=source_guideline_path,
        adapted_guideline_path=adapted_guideline_path,
        reuse_active_guideline=reuse_active_guideline,
    )
    progress["active_guideline_path"] = active_guideline_path
    progress["active_domain"] = target_domain
    save_progress(progress_file_path, progress)

    mode_name, enable_guideline_update, run_start, run_end = choose_run_mode(
        start_idx=start_idx,
        total_dataset_size=total_dataset_size,
    )
    run_samples = all_sample_reviews[run_start:run_end]
    actual_run_count = len(run_samples)

    print(
        f"\n[RUN] Mode: {mode_name} | Samples: {actual_run_count} "
        f"(dataset rows {run_start + 1} to {run_end})"
    )
    reset_current_run_logs(conflict_log_path, cause_data_path)

    print("\n[STEP 2] Build guideline vector DB (RAG)...")
    build_vector_database(file_path=active_guideline_path, persist_directory=db_directory)

    global_agreed, global_conflict = run_annotation_stage_v2(
        run_samples=run_samples,
        run_start=run_start,
        system_data_dir=system_data_dir,
    )

    print("\n" + "=" * 50)
    print(f"ANNOTATION SUMMARY ({actual_run_count} samples)")
    print(f"Agreed: {global_agreed} | Conflicts: {global_conflict}")
    print("=" * 50)

    resolved_conflict_count = run_conflict_stage_v2(
        conflict_log_path=conflict_log_path,
        enable_guideline_update=enable_guideline_update,
    )

    if enable_guideline_update and resolved_conflict_count > 0:
        print("\n[STEP 5] Guideline Agent + human-in-the-loop review...")
        process_all_causes(active_guideline_path=active_guideline_path, target_domain=target_domain)
    elif enable_guideline_update:
        print("\n[STEP 5] Skipped guideline update because K=0 (no conflicts/root-cause records).")
    else:
        print("\n[STEP 5] Skipped guideline update by annotation-only mode.")

    progress["last_processed_index"] = run_end
    progress["active_guideline_path"] = active_guideline_path
    progress["active_domain"] = target_domain
    save_progress(progress_file_path, progress)

    print("\n" + "*" * 60)
    print(f"Saved progress: completed up to sample {run_end}/{total_dataset_size}.")
    if enable_guideline_update:
        print("Guideline updates, if approved, are ready for the next run.")
    else:
        print("Root-cause analysis and guideline update were skipped.")
    print("*" * 60)

    if run_end < total_dataset_size:
        cont = input("\nRun another pipeline cycle now? (y/n): ").strip().lower()
        if cont == "y":
            print("\n" + "~" * 60)
            run_workflow_v2(reuse_active_guideline=True)
        else:
            print("Stopped. Run this script again to continue from saved progress.")


if __name__ == "__main__":
    run_workflow_v2()
