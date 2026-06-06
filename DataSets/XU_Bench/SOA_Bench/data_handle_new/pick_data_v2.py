import os
import pickle
import re
from tqdm import tqdm

# 配置路径
INPUT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/processed_pkl"
OUTPUT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res"
TARGET_COUNT = 10  # 每个文件要挑选的样本数量

# COCO类别映射关系 (名称 -> ID)
CATEGORY_NAME_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79
}

def get_category_id_from_label(label_name):
    """从标签名称中提取类别ID"""
    # 从label_00_person格式中提取数字部分
    match = re.match(r'label_(\d+)_', label_name)
    if match:
        return int(match.group(1))
    return None

def convert_mask(normalized_mask):
    """
    将归一化的mask [x, y, width, height] 转换为
    512x512分辨率的 [X1, Y1, X2, Y2] 格式
    """
    x, y, width, height = normalized_mask
    
    # 转换为像素坐标
    x1 = int(x * 512)
    y1 = int(y * 512)
    w = int(width * 512)
    h = int(height * 512)
    
    # 计算右下角坐标
    x2 = x1 + w
    y2 = y1 + h
    
    # 确保坐标在有效范围内
    x1 = max(0, min(x1, 511))
    y1 = max(0, min(y1, 511))
    x2 = max(0, min(x2, 511))
    y2 = max(0, min(y2, 511))
    
    return [x1, y1, x2, y2]

def process_file(file_path, label_name, target_category_id):
    """处理单个PKL文件，挑选符合条件的数据"""
    try:
        # 读取PKL文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            data = [data]
        
        filtered_data = []
        
        # 筛选符合条件的数据
        for item in data:
            if not isinstance(item, dict) or "prompt" not in item:
                continue
            
            # 统计当前类别出现的次数
            count = 0
            for key in item:
                if key != "prompt" and isinstance(item[key], dict):
                    desc = item[key].get("description", "").lower()
                    # 获取描述对应的类别ID
                    item_category_id = CATEGORY_NAME_TO_ID.get(desc, -1)
                    if item_category_id == target_category_id:
                        count += 1
            
            # 检查是否在1~5个之间
            if 1 <= count <= 10:
                # 转换mask格式
                converted_item = {"prompt": item["prompt"]}
                for key in item:
                    if key != "prompt" and isinstance(item[key], dict):
                        mask = item[key].get("mask", [0, 0, 0, 0])
                        converted_mask = convert_mask(mask)
                        converted_item[key] = {
                            "description": item[key].get("description", ""),
                            "mask": converted_mask,
                            "category_id": item[key].get("category_id", -1)
                        }
                if converted_item not in filtered_data:
                    filtered_data.append(converted_item)
                
                # 达到目标数量则停止
                if len(filtered_data) >= TARGET_COUNT:
                    break
        
        return filtered_data
        
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
        # 提取标签名称（不含.pkl扩展名）
        label_name = os.path.splitext(pkl_file)[0]
        file_path = os.path.join(INPUT_DIR, pkl_file)
        
        # 获取目标类别ID
        target_category_id = get_category_id_from_label(label_name)
        if target_category_id is None:
            print(f"无法从 {label_name} 中提取类别ID，跳过")
            continue
        
        # 处理文件并筛选数据
        filtered_data = process_file(file_path, label_name, target_category_id)
        
        if filtered_data:
            print(f"{label_name}: 筛选出 {len(filtered_data)} 条符合条件的数据")
            # 保存筛选后的数据
            output_path = os.path.join(OUTPUT_DIR, pkl_file)
            with open(output_path, 'wb') as f:
                pickle.dump(filtered_data, f)
        else:
            print(f"{label_name}: 未找到符合条件的数据")
    
    print(f"所有处理结果已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_files()
    