from chatGPT import read_txt_hrs, load_gt, load_box, save_img, read_csv, Pharse2idx_2, process_box_phrase, format_box, draw_box_2, generate_box_qwen
import pickle  # 导入pickle库用于序列化对象
from typing import List, Dict, Tuple

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

if __name__ == "__main__":
    _ ,prompts = load_gt('/home/sxm/HomeWorkSpace/testConnection/DataSets/HRS_Bench_Prompts/visual_writing_prompts.csv')
    # 存储所有样本
    dataset = []
    # box_data = load_box("data_evaluate_LLM/gpt_generated_box/counting.p")
    for i, prompt in enumerate(prompts):
        # if i != 2998:
        #     continue
        text = prompt[0]
        print('text:', text)
        o_names, o_boxes = generate_box_qwen(text)
        print('o_names', o_names)
        print('o_boxes', o_boxes)

        # 检查是否成功生成
        if not o_names or not o_boxes:
            print(f"⚠️  Warning: No boxes generated for prompt {i}")
            continue
        
        # 转换为标准格式
        sample = convert_to_dataset_format(text, o_names, o_boxes)
        print('sample:', sample)
        dataset.append(sample)

        # break
    
    # ✅ 修改：保存为 pickle 文件
    output_path = "/home/sxm/HomeWorkSpace/testConnection/DataSets/XU_Bench/Visual_Text_Bench/converted_dataset.p"
    with open(output_path, 'wb') as f:  # 注意这里使用了'wb'模式打开文件，表示二进制写入
        pickle.dump(dataset, f)

    print(f"\n✅ Successfully converted {len(dataset)} samples to {output_path}")
