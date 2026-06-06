import os
import pickle
import random
from tqdm import tqdm

def convert_mask(normalized_mask, resolution=512):
    """
    将归一化的[x, y, width, height]格式转换为
    512x512分辨率下的[X1, Y1, X2, Y2]格式（左上角和右下角坐标）
    """
    x, y, width, height = normalized_mask
    
    # 转换为像素坐标
    x1 = int(round(x * resolution))
    y1 = int(round(y * resolution))
    w = int(round(width * resolution))
    h = int(round(height * resolution))
    
    # 计算右下角坐标
    x2 = x1 + w
    y2 = y1 + h
    
    # 确保坐标在0到resolution范围内
    x1 = max(0, min(resolution, x1))
    y1 = max(0, min(resolution, y1))
    x2 = max(0, min(resolution, x2))
    y2 = max(0, min(resolution, y2))
    
    return [x1, y1, x2, y2]

def process_files(input_dir, output_dir, sample_size=10):
    """
    处理所有pkl文件，每个文件随机抽取指定数量的数据
    并转换mask格式
    """
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
        
        # 确保有足够的数据可供抽取
        if len(data_list) < sample_size:
            print(f"警告: {filename} 只有 {len(data_list)} 条数据，少于需要抽取的 {sample_size} 条")
            # 抽取所有可用数据
            sampled_data = data_list
        else:
            # 随机抽取指定数量的数据
            sampled_data = random.sample(data_list, sample_size)
        
        # 处理每条数据，转换mask格式
        processed_data = []
        for item in sampled_data:
            processed_item = {
                'prompt': item['prompt']
            }
            
            # 处理每个mask
            for key, value in item.items():
                if key != 'prompt' and isinstance(value, dict) and 'mask' in value:
                    # 转换mask格式
                    converted_mask = convert_mask(value['mask'])
                    processed_item[key] = {
                        'description': value['description'],
                        'mask': converted_mask,
                        'category_id': value['category_id']
                    }
            
            processed_data.append(processed_item)
        
        # 保存处理后的数据
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
    
    print(f"所有文件处理完成，结果保存在 {output_dir}")

def main():
    # 配置文件路径
    input_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/processed_pkl"
    output_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res"  # 新文件夹路径
    sample_size = 10  # 每个文件抽取的数据条数
    
    # 处理文件
    process_files(input_dir, output_dir, sample_size)

if __name__ == "__main__":
    main()
