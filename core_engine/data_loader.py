import os
import json


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_and_assign_ids(file_path):
    samples_with_ids = []
    
    if not os.path.exists(file_path):
        print(f"[CẢNH BÁO]: Không tìm thấy file dataset tại {file_path}")
        return samples_with_ids
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        counter = 1 # Biến đếm để tạo ID tăng dần
        
        for i in range(len(lines)):
            if lines[i].startswith('#'):
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
        print(f"Lỗi khi lưu file: {e}")

