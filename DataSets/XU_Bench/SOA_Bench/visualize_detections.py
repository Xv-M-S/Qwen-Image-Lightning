import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm

# 配置参数
ROOT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGINAME"  # 替换为您的根目录路径
OUTPUT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGINAME_VISUALIZATION"  # 可视化结果保存目录
# ROOT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/extracted_images"  # 替换为您的根目录路径
# OUTPUT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/extracted_images_VISUALIZATION"  # 可视化结果保存目录
# PKL_FILENAME = "results.pkl"  # PKL文件名，根据实际情况修改

# COCO风格的类别映射关系
LABEL_MAP = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

# 生成颜色映射，为每个类别分配不同的颜色
COLOR_MAP = {}
for i in range(len(LABEL_MAP)):
    # 使用HSV颜色空间生成均匀分布的颜色
    hue = i / len(LABEL_MAP)
    rgb = cv2.cvtColor(np.array([[[hue * 179, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    COLOR_MAP[i] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

def load_pkl_data(pkl_path):
    """读取PKL文件数据"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        # 确保数据是列表格式
        if isinstance(data, dict):
            return [data]
        return data
    except Exception as e:
        print(f"读取PKL文件失败 {pkl_path}: {str(e)}")
        return None

def visualize_detections(image_path, detections, output_path):
    """在图片上绘制检测结果并保存"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}")
        return False
    
    height, width = image.shape[:2]
    
    # 绘制每个检测结果
    for cls_id, bbox in zip(detections[1], detections[2]):
        cls_id = cls_id - 1
        # 解析边界框 (归一化坐标: x_center, y_center, w, h)
        x_center, y_center, bbox_w, bbox_h = bbox
        
        # 转换为像素坐标
        x1 = int((x_center - bbox_w / 2) * width)
        y1 = int((y_center - bbox_h / 2) * height)
        x2 = int((x_center + bbox_w / 2) * width)
        y2 = int((y_center + bbox_h / 2) * height)
        
        # 确保边界框在图片范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        
        # 获取类别名称和颜色
        cls_name = LABEL_MAP.get(cls_id, f"unknown({cls_id})")
        color = COLOR_MAP.get(cls_id % len(COLOR_MAP), (0, 0, 255))  # 未知类别用红色
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签
        label = f"{cls_name} ({cls_id})"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(image, (x1, y1_label - label_size[1] - 10),
                      (x1 + label_size[0], y1_label + base_line - 10),
                      color, -1)
        cv2.putText(image, label, (x1, y1_label - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存结果图片
    cv2.imwrite(output_path, image)
    return True

def process_all_labels():
    """处理所有类别文件夹"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有类别文件夹
    label_folders = [f for f in os.listdir(ROOT_DIR) 
                    if os.path.isdir(os.path.join(ROOT_DIR, f)) 
                    and f.startswith("label_")]
    
    # 处理每个类别文件夹
    for folder in tqdm(label_folders, desc="处理类别文件夹"):
        folder_path = os.path.join(ROOT_DIR, folder)
        output_folder = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # 查找PKL文件
        PKL_FILENAME = f"{folder}_ground_truth.pkl"

        # BASE_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/processed_pkl"
        # PKL_FILENAME = f"{BASE_PATH}/{folder}.pkl"

        pkl_path = os.path.join(folder_path, PKL_FILENAME)
        if not os.path.exists(pkl_path):
            print(f"在 {folder} 中未找到PKL文件 {PKL_FILENAME}，跳过该文件夹")
            continue
        
        # 加载PKL数据
        pkl_data = load_pkl_data(pkl_path)
        if not pkl_data:
            continue
        
        # 处理PKL中的每张图片
        for img_dict in pkl_data:
            for img_name, detections in img_dict.items():
                # 构建图片路径
                img_path = os.path.join(folder_path, img_name.split(".")[0] + "_generated_image.png")
                if not os.path.exists(img_path):
                    print(f"图片 {img_path} 不存在，跳过")
                    continue
                
                # 构建输出路径
                output_path = os.path.join(output_folder, img_name)
                
                # 可视化并保存
                visualize_detections(img_path, detections, output_path)
    
    print(f"所有可视化结果已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_labels()
