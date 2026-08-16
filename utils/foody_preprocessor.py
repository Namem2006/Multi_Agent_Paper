import os


def prepare_foody_dataset(source_path: str, output_path: str) -> int:
    """
    Convert a raw Foody text file into the line-based format expected by the pipeline.

    Each non-empty line becomes one sample:
    #0001
    review text

    Returns the number of samples written.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {source_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = []
    with open(source_path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            samples.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            f.write(f"#{idx:04d}\n")
            f.write(sample + "\n\n")

    print(f"[FOODY PREPROCESS] Đã tạo {len(samples)} sample tại: {output_path}")
    return len(samples)


def ensure_foody_dataset(source_path: str, output_path: str) -> int:
    """
    Rebuild the prepared file when it is missing or older than the raw source.
    """
    if not os.path.exists(output_path):
        return prepare_foody_dataset(source_path, output_path)

    source_mtime = os.path.getmtime(source_path)
    output_mtime = os.path.getmtime(output_path)
    if source_mtime > output_mtime:
        return prepare_foody_dataset(source_path, output_path)

    return 0