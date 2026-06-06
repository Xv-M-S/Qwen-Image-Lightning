import os
import json
import pickle
import shutil
from tqdm import tqdm

# 配置路径
PKL_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/captions"
CAPTION_TRAIN_JSON = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_train2014.json"
CAPTION_VAL_JSON = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_val2014.json"
INSTANCES_TRAIN_JSON = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_train2014.json"
INSTANCES_VAL_JSON = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_val2014.json"

# 输出路径
OUTPUT_PKL_DIR = "./processed_pkl"
OUTPUT_IMAGE_DIR = "./extracted_images"

# COCO图片路径 (假设图片路径，需要根据实际情况修改)
COCO_TRAIN_IMAGE_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/train2014"
COCO_VAL_IMAGE_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/val2014"




# golbal_id_map
global_id_map_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mappings/image_id_to_caption.json"
golbal_caption_map_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mappings/caption_to_image_id.json"

# 类别标签列表
coco80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# 生成 label_xx_xxx 格式
LABEL_FOLDERS = [
    f"label_{idx:02d}_{name.replace(' ', '_')}" 
    for idx, name in enumerate(coco80)
]
# LABEL_FOLDERS = [
#     "label_00_person", "label_01_bicycle", "label_02_car", "label_03_motorcycle",
#     "label_04_plane", "label_05_bus", "label_06_train", "label_07_truck",
#     "label_08_boat", "label_09_trafficlight", "label_10_hydrant", "label_11_stopsign",
#     "label_12_parkingmeter", "label_13_bench", "label_14_bird", "label_15_cat",
#     "label_16_dog", "label_17_horse", "label_18_sheep", "label_19_cow",
#     "label_20_elephant", "label_21_bear", "label_22_zebra", "label_23_giraffe",
#     "label_24_backpack", "label_25_umbrella", "label_26_handbag"
# ]

def load_json_data(json_path):
    """加载JSON文件数据"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败 {json_path}: {str(e)}")
        return None

def build_caption_lookup(train_json, val_json):
    """构建image_id到caption的映射"""
    caption_lookup = {}
    
    # 处理训练集
    if train_json and "annotations" in train_json:
        for ann in train_json["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            if image_id not in caption_lookup:
                caption_lookup[image_id] = []
            caption_lookup[image_id].append(caption)
    
    # 处理验证集
    if val_json and "annotations" in val_json:
        for ann in val_json["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            if image_id not in caption_lookup:
                caption_lookup[image_id] = []
            caption_lookup[image_id].append(caption)
    
    return caption_lookup

def build_instances_lookup(train_json, val_json):
    """构建image_id到实例信息的映射，包含归一化的边界框"""
    instances_lookup = {}
    
    # 首先获取类别映射
    categories = {}
    if train_json and "categories" in train_json:
        for cat in train_json["categories"]:
            categories[cat["id"]] = cat["name"]
    if val_json and "categories" in val_json:
        for cat in val_json["categories"]:
            categories[cat["id"]] = cat["name"]
    
    # 处理训练集实例
    if train_json and "annotations" in train_json and "images" in train_json:
        # 先构建image_id到图片尺寸的映射
        image_sizes = {img["id"]: (img["width"], img["height"]) for img in train_json["images"]}
        
        for ann in train_json["annotations"]:
            image_id = ann["image_id"]
            if image_id not in image_sizes:
                continue
                
            width, height = image_sizes[image_id]
            # COCO边界框格式: [x, y, width, height] (像素坐标)
            bbox = ann["bbox"]
            x, y, bbox_width, bbox_height = bbox
            
            # 归一化边界框
            norm_x = x / width
            norm_y = y / height
            norm_width = bbox_width / width
            norm_height = bbox_height / height
            
            category_id = ann["category_id"]
            category_name = categories.get(category_id, f"unknown_{category_id}")
            
            if image_id not in instances_lookup:
                instances_lookup[image_id] = []
            
            instances_lookup[image_id].append({
                "description": category_name,
                "mask": [norm_x, norm_y, norm_width, norm_height]  # 归一化格式
            })
    
    # 处理验证集实例
    if val_json and "annotations" in val_json and "images" in val_json:
        # 先构建image_id到图片尺寸的映射
        image_sizes = {img["id"]: (img["width"], img["height"]) for img in val_json["images"]}
        
        for ann in val_json["annotations"]:
            image_id = ann["image_id"]
            if image_id not in image_sizes:
                continue
                
            width, height = image_sizes[image_id]
            # COCO边界框格式: [x, y, width, height] (像素坐标)
            bbox = ann["bbox"]
            x, y, bbox_width, bbox_height = bbox
            
            # 归一化边界框
            norm_x = x / width
            norm_y = y / height
            norm_width = bbox_width / width
            norm_height = bbox_height / height
            
            category_id = ann["category_id"]
            category_name = categories.get(category_id, f"unknown_{category_id}")
            
            if image_id not in instances_lookup:
                instances_lookup[image_id] = []
            
            instances_lookup[image_id].append({
                "description": category_name,
                "mask": [norm_x, norm_y, norm_width, norm_height],  # 归一化格式
                "category_id": category_id
            })
    
    return instances_lookup

def get_image_path(image_id, train_images, val_images):
    """获取图片路径"""
    # 检查是否在训练集
    for img in train_images:
        if img["id"] == image_id:
            return os.path.join(COCO_TRAIN_IMAGE_DIR, img["file_name"])
    
    # 检查是否在验证集
    for img in val_images:
        if img["id"] == image_id:
            return os.path.join(COCO_VAL_IMAGE_DIR, img["file_name"])
    
    return None

def process_pkl_files():
    """处理所有PKL文件"""
    # 创建输出目录
    os.makedirs(OUTPUT_PKL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    
    # 创建所有类别图片文件夹
    for folder in LABEL_FOLDERS:
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, folder), exist_ok=True)
    
    # 加载COCO标注数据
    print("加载COCO标注数据...")
    caption_train_data = load_json_data(CAPTION_TRAIN_JSON)
    caption_val_data = load_json_data(CAPTION_VAL_JSON)
    instances_train_data = load_json_data(INSTANCES_TRAIN_JSON)
    instances_val_data = load_json_data(INSTANCES_VAL_JSON)

    global_id_map = load_json_data(global_id_map_path)
    global_caption_map = load_json_data(golbal_caption_map_path)
    
    # 构建查询表
    print("构建查询表...")
    caption_lookup = build_caption_lookup(caption_train_data, caption_val_data)
    instances_lookup = build_instances_lookup(instances_train_data, instances_val_data)
    
    # 获取所有图片信息
    train_images = instances_train_data.get("images", []) if instances_train_data else []
    val_images = instances_val_data.get("images", []) if instances_val_data else []
    
    # 获取所有PKL文件
    pkl_files = [f for f in os.listdir(PKL_DIR) if f.endswith(".pkl")]
    
    # 处理每个PKL文件
    for pkl_file in tqdm(pkl_files, desc="处理PKL文件"):
        pkl_path = os.path.join(PKL_DIR, pkl_file)
        output_pkl_path = os.path.join(OUTPUT_PKL_DIR, pkl_file)
        
        try:
            # 读取PKL文件
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # 确保数据是列表格式
            if not isinstance(data, list):
                data = [data]
            
            processed_data = []
            
            # 处理每个条目
            for item in data:
                image_id = item.get("image_id")
                caption = item.get("caption")
                
                if not image_id or not caption:
                    print(f"PKL文件 {pkl_file} 中缺少image_id或caption，跳过该条目")
                    continue

                if str(image_id) not in global_id_map or caption not in global_caption_map:
                    print(f"警告: image_id {image_id} 或 caption 在全局映射中未找到，跳过该条目")
                    continue

                
                
                # 1. 比对caption
                if image_id in caption_lookup:
                    coco_captions = caption_lookup[image_id]
                    # 检查当前caption是否在COCO的caption列表中
                    if not any(caption.strip().lower() == coco_cap.strip().lower() for coco_cap in coco_captions):
                        print(f"警告: 图片ID {image_id} 的caption不匹配")
                        print(f"  PKL中的caption: {caption}")
                        print(f"  COCO中的caption: {coco_captions[0]}")
                else:
                    print(f"警告: 在COCO标注中未找到图片ID {image_id} 的caption")
                
                # 2. 获取layout信息
                layout = {}
                if image_id in instances_lookup:
                    instances = instances_lookup[image_id]
                    for i, instance in enumerate(instances):
                        layout[str(i)] = {
                            "description": instance["description"],
                            "mask": instance["mask"],
                            "category_id": instance["category_id"]
                        }
                
                # 3. 构建输出格式
                output_item = {
                    "prompt": caption,
                    **layout
                }
                processed_data.append(output_item)
                
                # 4. 复制图片到对应目录
                # 提取标签类别 (如从label_01_bicycle.pkl中提取label_01_bicycle)
                label_folder = os.path.splitext(pkl_file)[0]
                if label_folder in LABEL_FOLDERS:
                    image_path = get_image_path(image_id, train_images, val_images)
                    if image_path and os.path.exists(image_path):
                        dest_folder = os.path.join(OUTPUT_IMAGE_DIR, label_folder)
                        dest_path = os.path.join(dest_folder, os.path.basename(image_path))
                        # 避免重复复制
                        if not os.path.exists(dest_path):
                            shutil.copy2(image_path, dest_path)
            
            # 保存处理后的PKL文件
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(processed_data, f)
                
        except Exception as e:
            print(f"处理PKL文件 {pkl_file} 时出错: {str(e)}")
            continue
    
    print(f"所有处理结果已保存至 {OUTPUT_PKL_DIR}")
    print(f"提取的图片已保存至 {OUTPUT_IMAGE_DIR}")

if __name__ == "__main__":
    process_pkl_files()
