import os
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_and_assign_ids(file_path):
    samples_with_ids = []

    if not os.path.exists(file_path):
        print(f"[CẢNH BÁO]: Không tìm thấy file dataset tại {file_path}")
        return samples_with_ids

    # SỬA LỖI: Dùng utf-8-sig để loại bỏ ký tự BOM ẩn ( \ufeff ) ở dòng đầu tiên của file text
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

        counter = 1 # Biến đếm để tạo ID tăng dần

        for i in range(len(lines)):
            # Thêm .strip() để đảm bảo không bị ảnh hưởng bởi khoảng trắng thừa
            if lines[i].strip().startswith('#'):
                # Dòng tiếp theo chứa nội dung review
                if i + 1 < len(lines):
                    content = lines[i+1].strip()

                    # Bỏ qua dòng trống và dòng chứa nhãn (bắt đầu bằng '{')
                    if content and not content.startswith('{'):
                        # Tạo ID định dạng REV_0001, REV_0002, ...
                        review_id = f"#{counter:04d}"

                        # Đóng gói thành dictionary
                        samples_with_ids.append({
                            "id": review_id,
                            "text": content
                        })

                        counter += 1

    return samples_with_ids

def save_to_json(data, output_filepath):
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu {len(data)} câu review có kèm ID vào file: {output_filepath}")
    except Exception as e:
        print(f"[LỖI] Không thể lưu file JSON: {e}")

# Dùng để test file khi chạy độc lập
if __name__ == "__main__":
    # Test với file thực tế
    test_file = os.path.join(os.path.dirname(ROOT_DIR), "data", "1-VLSP2018-SA-Restaurant-train (7-3-2018).txt")
    output_test = os.path.join(os.path.dirname(ROOT_DIR), "system_data", "extracted_samples.json")

    extracted_data = extract_and_assign_ids(test_file)
    print(f"Đã trích xuất được {len(extracted_data)} câu.")
    if extracted_data:
        print("Câu đầu tiên:", extracted_data[0])