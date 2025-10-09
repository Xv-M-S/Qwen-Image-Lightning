import json
import os

def create_caption_maps(input_json_path, output_id_to_caption, output_caption_to_id):
    """
    从输入JSON文件提取id和caption，生成双向映射并保存为JSON
    
    参数:
        input_json_path: 输入的tifa_v1.0_text_inputs.json路径
        output_id_to_caption: id到caption映射的输出JSON路径
        output_caption_to_id: caption到id映射的输出JSON路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"输入文件不存在: {input_json_path}")
    
    # 读取输入JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化两个映射字典
    id_to_caption = {}
    caption_to_id = {}
    
    # 遍历数据构建映射
    for item in data:
        # 检查必要字段是否存在
        if "id" not in item or "caption" not in item:
            print(f"警告: 跳过缺少id或caption的条目: {item}")
            continue
        
        item_id = item["id"]
        item_caption = item["caption"].strip()  # 去除首尾空格
        
        # 构建id到caption的映射
        id_to_caption[item_id] = item_caption
        
        # 构建caption到id的映射
        # 注意：如果存在相同caption，后面的会覆盖前面的
        if item_caption in caption_to_id:
            print(f"警告: 发现重复caption，将覆盖之前的id: {item_caption}")
        caption_to_id[item_caption] = item_id
    
    # 创建输出目录（如果不存在）
    for output_path in [output_id_to_caption, output_caption_to_id]:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # 保存id到caption的映射
    with open(output_id_to_caption, 'w', encoding='utf-8') as f:
        json.dump(id_to_caption, f, indent=2, ensure_ascii=False)
    
    # 保存caption到id的映射
    with open(output_caption_to_id, 'w', encoding='utf-8') as f:
        json.dump(caption_to_id, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成:")
    print(f"id到caption映射已保存至: {output_id_to_caption}")
    print(f"caption到id映射已保存至: {output_caption_to_id}")
    print(f"共处理 {len(id_to_caption)} 条有效数据")

if __name__ == "__main__":
    # 配置文件路径
    INPUT_JSON = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/tifa_v1.0_text_inputs.json"
    
    # 输出文件路径（可根据需要修改）
    OUTPUT_ID_TO_CAPTION = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/id_to_caption.json"
    OUTPUT_CAPTION_TO_ID = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/caption_to_id.json"
    
    # 执行映射生成
    create_caption_maps(INPUT_JSON, OUTPUT_ID_TO_CAPTION, OUTPUT_CAPTION_TO_ID)
    