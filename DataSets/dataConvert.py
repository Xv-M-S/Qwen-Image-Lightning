import pickle

def convert_dataset(data):
    """
    将原始数据转换为目标格式，自动跳过格式错误的 bbox。
    """
    converted = {}
    
    for prompt, (labels, bboxes) in data.items():
        entry = {"prompt": prompt}
        valid_count = 0
        
        for idx, (label, bbox) in enumerate(zip(labels, bboxes)):
            # 检查 bbox 是否为 4 元组 (x1, y1, x2, y2)
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                print(f"⚠️  跳过无效 bbox (index {idx}) in prompt: '{prompt[:30]}...'")
                print(f"    label: {label}, bbox: {bbox}")
                continue  # 跳过非法 bbox
            
            try:
                x1, y1, x2, y2 = map(int, bbox)
                entry[str(valid_count)] = {
                    "description": label,
                    "mask": [x1, y1, x2, y2]
                }
                valid_count += 1
            except (ValueError, TypeError) as e:
                print(f"❌ 无法解析 bbox {bbox}，跳过 (prompt: '{prompt[:30]}...')")
                continue
        
        # 确保至少有一个有效对象
        if valid_count == 0:
            print(f"❌ 提示 '{prompt}' 没有有效 bbox，已忽略。")
        else:
            # 使用 prompt 的简化版本作为 key
            key = f"{prompt[:50].replace(' ', '_').replace(',', '').replace('.', '')}_{valid_count}"
            converted[key] = entry

    return converted

if __name__ == "__main__":
    import os
    # case 1
    # ppath = "/home/sxm/HomeWorkSpace/testConnection/DataSets/data_evaluate_LLM/gpt_generated_box_drawbench"
    # outpath = "/home/sxm/HomeWorkSpace/testConnection/DataSets/XU_Bench/LLM_Bench"
    # case 2
    ppath = "/home/sxm/HomeWorkSpace/testConnection/DataSets/data_evaluate_LLM/gpt_generated_box_drawbench"
    outpath = "/home/sxm/HomeWorkSpace/testConnection/DataSets/XU_Bench/drawBench"
    for file in os.listdir(ppath):
        if file.endswith(".p"):
            with open(os.path.join(ppath, file), 'rb') as f:
                raw_data = pickle.load(f)
            converted_data = convert_dataset(raw_data)
            with open(os.path.join(outpath, file), 'wb') as f:
                pickle.dump(converted_data, f)

    single_test = True
    if single_test:
        # 读取原始 .p 或 .pkl 文件
        with open('/home/sxm/HomeWorkSpace/testConnection/DataSets/data_evaluate_LLM/gpt_generated_box_drawbench/counting.p', 'rb') as f:
            raw_data = pickle.load(f)

        # 转换
        converted_data = convert_dataset(raw_data)
        for k, v in list(converted_data.items())[:3]:  # 仅打印前3条以示例
            print(f"{k}: {v}")
        print(f"转换后数据条目数: {len(converted_data)}")

        # 保存为新的 .p 文件
        # with open('/home/sxm/flux-workspace/layout-to-image-zhuanlan/attention-refocusing/sxmTest/converted_dataset.p', 'wb') as f:
        #     pickle.dump(converted_data, f)