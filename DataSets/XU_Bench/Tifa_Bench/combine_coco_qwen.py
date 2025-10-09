import pickle
import json
import os

def load_pickle_file(file_path):
    """加载pickle文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle文件不存在: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_caption_id_map(json_path):
    """加载caption到id的映射文件"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON映射文件不存在: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def combine_datasets(coco_data, tifa_data, caption_to_id):
    """
    合并数据集，将tifaData中独有的数据添加到coco_data中
    
    参数:
        coco_data: 来自coco_prompt_layout.p的数据
        tifa_data: 来自tifaData.p的数据
        caption_to_id: 从caption_to_id.json加载的映射
    返回:
        合并后的数据集
    """
    # 构建coco数据中已有的prompt集合（用于去重）
    # 先通过caption_to_id映射找到每个prompt对应的id，再用id判断唯一性
    coco_ids = set()
    
    print(f"tifaData.p中包含 {len(tifa_data)} 个数据")
    print(f"coco_prompt_layout.p中包含 {len(coco_data)} 个数据")

    for item in coco_data:
        prompt = item.get('prompt', '').strip()
        # 通过prompt找到对应的id
        item_id = caption_to_id.get(prompt)
        print(f"prompt: {prompt}, id: {item_id}")
        if item_id:
            coco_ids.add(item_id)
    
    print(f"coco_prompt_layout.p中包含 {len(coco_ids)} 个唯一数据")
    
    # 收集tifaData中独有的数据
    tifa_unique = []
    for item in tifa_data:
        prompt = item.get('prompt', '').strip()
        item_id = caption_to_id.get(prompt)
        
        # 如果id不存在于coco数据中，则认为是独有数据
        if item_id and item_id not in coco_ids:
            tifa_unique.append(item)
            coco_ids.add(item_id)  # 避免重复添加
    
    print(f"tifaData.p中包含 {len(tifa_unique)} 个coco中没有的独有数据")
    
    # 合并数据：coco数据 + tifa独有数据
    combined_data = coco_data + tifa_unique
    print(f"合并后的数据总数: {len(combined_data)}")
    
    return combined_data

def save_combined_data(combined_data, output_path):
    """保存合并后的数据到新的pickle文件"""
    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(combined_data, f)
    print(f"合并后的数据已保存至: {output_path}")

if __name__ == "__main__":
    # 配置文件路径
    COCO_PICKLE_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/coco_caption_layout.pkl"  # coco_prompt_layout.p的路径
    TIFA_PICKLE_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifaData.p"            # tifaData.p的路径
    CAPTION_ID_JSON_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/caption_to_id.json"
    OUTPUT_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/qwen_coco_combined.p"       # 输出文件路径
    
    try:
        # 加载数据
        print("正在加载数据...")
        coco_data = load_pickle_file(COCO_PICKLE_PATH)
        tifa_data = load_pickle_file(TIFA_PICKLE_PATH)
        caption_to_id = load_caption_id_map(CAPTION_ID_JSON_PATH)

        # print(f"成功加载caption_to_id映射，共 {len(caption_to_id)} 条记录")
        # print(caption_to_id)
        
        # 验证数据格式（确保是列表）
        if not isinstance(coco_data, list) or not isinstance(tifa_data, list):
            raise ValueError("pickle文件内容必须是列表格式")
        
        # 合并数据
        print("正在合并数据...")
        combined_data = combine_datasets(coco_data, tifa_data, caption_to_id)
        
        # 保存结果
        save_combined_data(combined_data, OUTPUT_PATH)
        
    except Exception as e:
        print(f"处理过程出错: {str(e)}")
    