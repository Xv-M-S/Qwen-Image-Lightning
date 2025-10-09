import os
import json
import pickle
from tqdm import tqdm

def load_coco_captions(train_captions_path, val_captions_path):
    """加载COCO的caption标注数据"""
    print("加载COCO caption数据...")
    with open(train_captions_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_captions_path, 'r') as f:
        val_data = json.load(f)
    
    # 创建image_id到captions的映射
    caption_map = {}
    for data in [train_data, val_data]:
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in caption_map:
                caption_map[image_id] = []
            caption_map[image_id].append(ann['caption'])
    
    return caption_map

def load_coco_instances(train_instances_path, val_instances_path):
    """加载COCO的实例标注数据（用于获取layout信息）"""
    print("加载COCO instances数据...")
    with open(train_instances_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_instances_path, 'r') as f:
        val_data = json.load(f)
    
    # 创建category_id到category名称的映射
    categories = {}
    for cat in train_data['categories'] + val_data['categories']:
        categories[cat['id']] = cat['name']
    
    # 创建image_id到图像尺寸的映射
    image_sizes = {}
    for data in [train_data, val_data]:
        for img in data['images']:
            image_sizes[img['id']] = (img['width'], img['height'])
    
    # 创建image_id到实例的映射
    instances_map = {}
    for data in [train_data, val_data]:
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in instances_map:
                instances_map[image_id] = []
            
            # 获取图像尺寸用于归一化
            if image_id not in image_sizes:
                print(f"警告: 未找到image_id {image_id} 的尺寸信息，跳过该实例")
                continue
                
            img_width, img_height = image_sizes[image_id]
            
            # 提取边界框信息并转换为归一化的[x, y, width, height]格式
            bbox = ann['bbox']  # COCO原始格式是 [x, y, width, height]
            
            # 归一化 - 将坐标除以图像宽度/高度
            x = bbox[0] / img_width
            y = bbox[1] / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            
            # 确保值在0到1之间（处理可能的标注误差）
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            width = max(0, min(1 - x, width))
            height = max(0, min(1 - y, height))
            
            instances_map[image_id].append({
                'description': categories.get(ann['category_id'], f"category_{ann['category_id']}"),
                'mask': [x, y, width, height],  # 归一化格式
                'category_id' : ann['category_id']
            })
    
    return instances_map

def process_pkl_files(input_dir, output_dir, caption_map, instances_map):
    """处理所有pkl文件并生成新的标注文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有pkl文件
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个pkl文件，开始处理...")
    
    for filename in tqdm(pkl_files, desc="处理文件"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 读取pkl文件
        with open(input_path, 'rb') as f:
            data_list = pickle.load(f)
        
        processed_data = []
        
        for item in data_list:
            image_id = item['image_id']
            caption = item['caption']
            
            # 1. 比对caption
            if image_id in caption_map:
                coco_captions = caption_map[image_id]
                # 检查当前caption是否在COCO的captions中（忽略大小写）
                if not any(caption.lower() == coco_cap.lower() for coco_cap in coco_captions):
                    print(f"警告: image_id {image_id} 的caption不匹配")
                    print(f"  pkl中的caption: {caption}")
                    print(f"  COCO中的captions: {coco_captions}...")  # 只显示前两个
            else:
                print(f"警告: 在COCO caption数据中未找到image_id {image_id}")
            
            # 2. 获取layout信息
            layout_info = {}
            if image_id in instances_map:
                instances = instances_map[image_id]
                # 转换为所需的格式
                for i, instance in enumerate(instances[:10]):  # 限制最多10个实例，避免过多
                    layout_info[str(i)] = {
                        'description': instance['description'],
                        'mask': instance['mask'],  # 已归一化的格式 [x, y, width, height]
                        'category_id': instance['category_id']
                    }
            else:
                print(f"警告: 在COCO instances数据中未找到image_id {image_id}")
            
            # 3. 构建输出格式
            output_item = {
                'prompt': caption,** layout_info
            }
            processed_data.append(output_item)
        
        # 保存处理后的数据
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
    
    print(f"所有文件处理完成，结果保存在 {output_dir}")

def main():
    # 配置文件路径
    input_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/captions"
    output_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/processed_captions_with_layout"  # 新文件夹路径
    
    # COCO caption文件路径
    train_captions_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_train2014.json"
    val_captions_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_val2014.json"
    
    # COCO instances文件路径
    train_instances_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_train2014.json"
    val_instances_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_val2014.json"
    
    # 加载COCO数据
    caption_map = load_coco_captions(train_captions_path, val_captions_path)
    instances_map = load_coco_instances(train_instances_path, val_instances_path)
    
    # 处理pkl文件
    process_pkl_files(input_dir, output_dir, caption_map, instances_map)

if __name__ == "__main__":
    main()
