import os
import pickle
import cv2
import numpy as np
import re
from tqdm import tqdm

# 配置路径
IMAGE_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_images"  # 原始图片文件夹路径
PKL_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res_ground_truth"       # 标注文件(PKL)所在文件夹路径
OUTPUT_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/visualized_results"  # 可视化结果保存路径

# 图片文件名前缀（根据实际情况修改）
IMAGE_PREFIX = "COCO_val2014_"  # 例如：COCO_train2014_000000001234.jpg

# COCO类别映射关系 (category_id -> 类别名称)
COCO_LABEL_MAP = {
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
    79: "toothbrush",
    81: "unknown"  # 示例中出现的未知类别
}

# 生成每个类别的颜色（用于边界框）
def generate_colors(num_classes):
    """生成均匀分布的颜色用于不同类别"""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = cv2.cvtColor(np.array([[[hue * 179, 255, 255]]], dtype=np.uint8), 
                          cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return colors

# 为每个类别分配颜色
COLORS = generate_colors(len(COCO_LABEL_MAP))

def extract_image_id_from_filename(filename):
    """从图片文件名中提取image_id（如从'561256.png'中提取561256）"""
    match = re.match(r'^(\d+)\.\w+$', filename)
    if match:
        return int(match.group(1))
    return None

def get_image_path(label_name, image_id):
    """获取原始图片的路径"""
    image_name = f"{IMAGE_PREFIX}{image_id:012d}.jpg"
    return os.path.join(IMAGE_DIR, label_name, image_name)

def visualize_detections(image_path, categories, bboxes):
    """将检测结果（类别和边界框）绘制到图片上"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    # 绘制每个检测框
    for category_id, bbox in zip(categories, bboxes):
        # 解析边界框 (归一化坐标: x_center, y_center, width, height)
        x_center, y_center, bbox_width, bbox_height = bbox
        
        # 转换为像素坐标
        x1 = int((x_center - bbox_width / 2) * width)
        y1 = int((y_center - bbox_height / 2) * height)
        x2 = int((x_center + bbox_width / 2) * width)
        y2 = int((y_center + bbox_height / 2) * height)
        
        # 确保边界框在图片范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        
        # 获取类别名称
        category_name = COCO_LABEL_MAP.get(category_id - 1, f"unknown({category_id})")
        
        # 获取颜色（使用类别ID的哈希值确保同一类别颜色一致）
        color_idx = category_id % len(COLORS)
        color = COLORS[color_idx]
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签
        label = f"{category_name} ({category_id})"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 绘制标签背景
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1 + base_line - 10), color, -1)
        
        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def process_label(label_name):
    """处理指定类别的图片和标注"""
    # 创建输出目录
    output_label_dir = os.path.join(OUTPUT_DIR, label_name)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # PKL文件路径
    pkl_file = os.path.join(PKL_DIR, f"{label_name}_ground_truth.pkl")
    if not os.path.exists(pkl_file):
        print(f"警告: PKL文件 {pkl_file} 不存在，跳过该类别")
        return
    
    try:
        # 读取PKL文件
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            data = [data]
        
        # 处理每个图片的标注
        for item in data:
            if isinstance(item, dict):
                for filename, annotations in item.items():
                    # 提取image_id
                    image_id = extract_image_id_from_filename(filename)
                    if image_id is None:
                        print(f"警告: 无法从文件名 {filename} 中提取image_id，跳过")
                        continue
                    
                    # 解析标注信息 [未知字段, 类别ID列表, 边界框列表]
                    _, category_ids, bboxes = annotations
                    
                    # 获取原始图片路径
                    image_path = get_image_path(label_name, image_id)
                    if not os.path.exists(image_path):
                        print(f"警告: 图片 {image_path} 不存在，跳过")
                        continue
                    
                    # 可视化检测结果
                    visualized_image = visualize_detections(image_path, category_ids, bboxes)
                    if visualized_image is None:
                        print(f"警告: 无法处理图片 {image_path}，跳过")
                        continue
                    
                    # 保存可视化结果
                    output_image_path = os.path.join(output_label_dir, os.path.basename(image_path))
                    cv2.imwrite(output_image_path, visualized_image)
    
    except Exception as e:
        print(f"处理 {label_name} 时出错: {str(e)}")

def process_all_labels():
    """处理所有类别"""
    # 获取所有PKL文件并提取类别名称
    pkl_files = [f for f in os.listdir(PKL_DIR) if f.endswith(".pkl")]
    labels = [os.path.splitext(f)[0] for f in pkl_files]
    
    if not labels:
        print("未找到任何PKL文件")
        return
    
    # 创建输出根目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理每个类别
    for label in tqdm(labels, desc="处理类别"):
        label = label.replace("_ground_truth", "")
        process_label(label)
    
    print(f"所有可视化结果已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_labels()
    