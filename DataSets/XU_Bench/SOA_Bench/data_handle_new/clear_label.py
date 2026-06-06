import os
import pickle
import re
from tqdm import tqdm

# 配置路径
INPUT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res"
OUTPUT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res_clear_layout"

def extract_category_name_from_filename(filename):
    """从文件名中提取类别名称（如从label_00_person.pkl中提取person）"""
    match = re.match(r'label_\d+_(.+)\.pkl', filename)
    if match:
        return match.group(1)
    return None

def clean_layout_data(item, target_category):
    """清理布局数据，只保留目标类别的box信息"""
    # 保留prompt
    cleaned_item = {"prompt": item.get("prompt", "")}
    
    # 收集目标类别的box
    target_boxes = []
    for key, value in item.items():
        # 跳过prompt字段
        if key == "prompt":
            continue
            
        # 检查是否是目标类别
        if isinstance(value, dict) and value.get("description", "").lower() == target_category.lower():
            target_boxes.append((key, value))
    
    # 按原有序号重新编号
    for i, (original_key, box_data) in enumerate(target_boxes):
        cleaned_item[str(i)] = box_data
    
    return cleaned_item

def process_file(file_path, target_category):
    """处理单个PKL文件，清理无关的布局信息"""
    try:
        # 读取PKL文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            data = [data]
        
        cleaned_data = []
        
        # 处理每条数据
        for item in data:
            if isinstance(item, dict):
                cleaned_item = clean_layout_data(item, target_category)
                cleaned_data.append(cleaned_item)
        
        return cleaned_data
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return []

def process_all_files():
    """处理所有PKL文件"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有PKL文件
    pkl_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pkl")]
    
    if not pkl_files:
        print("未找到任何PKL文件")
        return
    
    # 处理每个PKL文件
    for pkl_file in tqdm(pkl_files, desc="处理文件"):
        # 提取目标类别名称
        target_category = extract_category_name_from_filename(pkl_file)
        if target_category is None:
            print(f"无法从 {pkl_file} 中提取类别名称，跳过")
            continue
        
        file_path = os.path.join(INPUT_DIR, pkl_file)
        
        # 处理文件并清理布局信息
        cleaned_data = process_file(file_path, target_category)
        
        if cleaned_data:
            print(f"{pkl_file}: 清理后保留 {len(cleaned_data)} 条数据")
            # 保存清理后的文件
            output_path = os.path.join(OUTPUT_DIR, pkl_file)
            with open(output_path, 'wb') as f:
                pickle.dump(cleaned_data, f)
        else:
            print(f"{pkl_file}: 清理后未保留任何数据")
    
    print(f"所有处理结果已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_files()
    