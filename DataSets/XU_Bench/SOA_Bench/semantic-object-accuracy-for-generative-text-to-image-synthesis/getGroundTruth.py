import os
import pickle
import json
from tqdm import tqdm

def convert_to_yolo_format(box, resolution=512):
    """
    将[x1, y1, x2, y2]格式转换为归一化的YOLO格式[x, y, width, height]
    x, y是边界框中心点坐标，width和height是边界框的宽和高
    所有值都相对于图像分辨率(512)进行归一化
    """
    x1, y1, x2, y2 = box
    
    # 计算中心点坐标
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    
    # 计算宽和高
    width = x2 - x1
    height = y2 - y1
    
    # 归一化
    x = x_center / resolution
    y = y_center / resolution
    w = width / resolution
    h = height / resolution
    
    # 确保值在0到1之间
    x = max(0, min(1, x))
    y = max(0, min(1, y))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    
    return [round(x, 6), round(y, 6), round(w, 6), round(h, 6)]

def process_files(input_dir, output_dir, caption_map_path):
    """
    处理所有pkl文件，转换为指定的ground truth格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载caption到image_id的映射表
    with open(caption_map_path, 'r', encoding='utf-8') as f:
        caption_to_id = json.load(f)
    
    # 获取所有pkl文件
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个pkl文件，开始处理...")
    
    for filename in tqdm(pkl_files, desc="处理文件"):
        # 构建输入输出路径
        input_path = os.path.join(input_dir, filename)
        # 生成新文件名，在原后缀前加上ground_truth
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_ground_truth.pkl"
        output_path = os.path.join(output_dir, output_filename)
        
        # 读取pkl文件
        with open(input_path, 'rb') as f:
            data_list = pickle.load(f)
        
        processed_data = []
        
        for item in data_list:
            prompt = item['prompt']
            
            # 获取对应的image_id
            if prompt in caption_to_id:
                image_id = caption_to_id[prompt]
                image_name = f"{image_id}.png"
            else:
                print(f"警告: 在映射表中未找到prompt '{prompt[:50]}...' 对应的id，将跳过该条目")
                continue
            
            # 提取类别ID和边界框
            category_ids = []
            boxes = []
            
            # 遍历所有目标（跳过'prompt'键）
            for key in item:
                if key != 'prompt' and isinstance(item[key], dict):
                    obj = item[key]
                    if 'category_id' in obj and 'mask' in obj:
                        # 转换边界框格式
                        yolo_box = convert_to_yolo_format(obj['mask'])
                        boxes.append(yolo_box)
                        category_ids.append(obj['category_id'])
            
            # 构建输出格式
            output_item = {
                image_name: [
                    [],  # 空列表
                    category_ids,
                    boxes
                ]
            }
            
            processed_data.append(output_item)
        
        # 保存处理后的数据
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
    
    print(f"所有文件处理完成，结果保存在 {output_dir}")

def main():
    # 配置文件路径
    input_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/sampled_captions_512res"
    output_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/ground_truth_results"
    caption_map_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mappings/caption_to_image_id.json"
    
    # 处理文件
    process_files(input_dir, output_dir, caption_map_path)

if __name__ == "__main__":
    main()
