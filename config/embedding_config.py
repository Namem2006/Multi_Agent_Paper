"""
Shared Embedding Model Configuration
Sử dụng singleton pattern để tránh load model nhiều lần
"""
import time
from langchain_huggingface import HuggingFaceEmbeddings

_embedding_instance = None

def get_embeddings(max_retries=3):
    """
    Lấy shared embedding instance (singleton pattern)
    Chỉ load model 1 lần duy nhất khi khởi động

    Args:
        max_retries: Số lần retry nếu gặp lỗi network
    """
    global _embedding_instance

    if _embedding_instance is None:
        print("[EMBEDDING] Đang load model multilingual-e5-large lần đầu tiên...")

        for attempt in range(max_retries):
            try:
                _embedding_instance = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-small",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print("[EMBEDDING] ✓ Đã load xong model. Sẽ dùng chung instance này.")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s...
                    print(f"[EMBEDDING] ⚠ Lỗi khi load model (lần {attempt + 1}/{max_retries}): {e}")
                    print(f"[EMBEDDING] Đang retry sau {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[EMBEDDING] ✗ Không thể load model sau {max_retries} lần thử.")
                    raise Exception(f"Failed to load embedding model: {e}")

    return _embedding_instance

def reset_embeddings():
    """Reset embedding instance (dùng khi cần reload model)"""
    global _embedding_instance
    _embedding_instance = None
