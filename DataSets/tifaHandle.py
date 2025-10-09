import json
from typing import List, Dict, Tuple
from chatGPT import read_txt_hrs, load_gt, load_box, save_img, read_csv, Pharse2idx_2, process_box_phrase, format_box, draw_box_2, generate_box_qwen
import pickle  # 导入pickle库用于序列化对象

def extract_captions(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有caption字段，形成列表
    captions = [item['caption'] for item in data]
    
    return captions

def convert_to_dataset_format(prompt: str, o_names: List[str], o_boxes: List[Tuple[int, int, int, int]]) -> Dict:
    """
    将单个样本转换为目标数据集格式
    """
    result = {
        "prompt": prompt
    }
    
    for idx, (name, box) in enumerate(zip(o_names, o_boxes)):
        result[str(idx)] = {
            "description": name.strip(),  # 去掉多余空格
            "mask": list(box)             # 转为 list，确保可序列化
        }
    
    return result

# 使用示例
if __name__ == "__main__":
    # 替换为你的JSON文件路径
    json_path = "/home/sxm/HomeWorkSpace/testConnection/DataSets/tifaData/sample_text_inputs.json"
    prompts = extract_captions(json_path)
    
    # 打印结果
    print(prompts)


    # 存储所有样本
    dataset = []
    # box_data = load_box("data_evaluate_LLM/gpt_generated_box/counting.p")
    for i, prompt in enumerate(prompts):
        text = prompt
        print('text:', text)

        try:
            o_names, o_boxes = generate_box_qwen(text)
            print('o_names', o_names)
            print('o_boxes', o_boxes)
        except:
            print('Error:', prompt)
            continue

        # 检查是否成功生成
        if not o_names or not o_boxes:
            print(f"⚠️  Warning: No boxes generated for prompt {i}")
            continue
        
        # 转换为标准格式
        try:
            sample = convert_to_dataset_format(text, o_names, o_boxes)
            print('sample:', sample)
            dataset.append(sample)
        except:
            print('Error:', prompt)
            continue
    
    # ✅ 修改：保存为 pickle 文件
    output_path = "/home/sxm/HomeWorkSpace/testConnection/DataSets/XU_Bench/TifaBench/tifaData.p"
    with open(output_path, 'wb') as f:  # 注意这里使用了'wb'模式打开文件，表示二进制写入
        pickle.dump(dataset, f)

    print(f"\n✅ Successfully converted {len(dataset)} samples to {output_path}")
