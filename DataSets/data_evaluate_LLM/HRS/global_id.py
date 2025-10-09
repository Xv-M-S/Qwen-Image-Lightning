import pandas as pd
import os
from typing import List
import json  # 引入JSON处理模块


def process_csv_to_json(csv_file_paths: List[str], output_json_path: str) -> None:
    """
    处理多个CSV文件，生成Key-Value映射并保存为JSON文件
    
    Args:
        csv_file_paths: CSV文件路径列表（如 ["data1.csv", "D:/data/data2.csv"]）
        output_json_path: 输出JSON文件的路径（如 "key_value_map.json"）
    """
    # 1. 初始化全局自增ID（从1开始）和Key-Value字典（JSON需支持的格式：字符串键+字符串值）
    global_id = 1
    key_value_map = {}
    id_prompt_map = {}
    
    # 2. 遍历每个CSV文件
    for csv_path in csv_file_paths:
        # 检查CSV文件是否存在
        if not os.path.exists(csv_path):
            print(f"警告：文件 {csv_path} 不存在，跳过处理")
            continue
        
        # 读取CSV文件（捕获读取异常，避免单个文件错误中断整体流程）
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"读取文件 {csv_path} 失败：{str(e)}，跳过处理")
            continue
        
        # 提取CSV文件名前缀（以"_"分割后的第一个单词，如 "data_2024.csv" → "data"）
        csv_filename = os.path.basename(csv_path)  # 获取文件名（不含路径）
        filename_prefix = csv_filename.split("_")[0]  # 分割后取第一个元素
        
        # 3. 遍历CSV每一行，生成Key和Value
        for row_idx, row in df.iterrows():
            # 3.1 生成Key：优先用synthetic_prompt，为空则用meta_prompt
            # 处理空值：排除NaN、空字符串、纯空格
            synthetic_prompt = str(row.get("synthetic_prompt", "")).strip()
            meta_prompt = str(row.get("meta_prompt", "")).strip()
            
            # print(f"synthetic_prompt: {synthetic_prompt} meta_prompt: {meta_prompt}" )
            # print(type(synthetic_prompt), type(meta_prompt))
            if not pd.isna(synthetic_prompt) and synthetic_prompt != "" and synthetic_prompt != "nan":
                current_key = synthetic_prompt
            elif not pd.isna(meta_prompt) and meta_prompt != "" and meta_prompt != "nan":
                current_key = meta_prompt
            else:
                # 两个字段均为空时跳过，打印警告便于排查
                print(f"警告：CSV {csv_path} 第 {row_idx+1} 行（索引{row_idx}）的两个prompt均为空，跳过")
                continue
            
            # 3.2 生成Value：文件名前缀 + 全局自增ID（如 "data_1"）
            current_value = f"{filename_prefix}_{global_id}"
            
            # 3.3 处理重复Key（后出现的覆盖前一个，或保留首次，可按需调整）
            if current_key in key_value_map:
                print(f"注意：Key '{current_key}' 已存在（原Value：{key_value_map[current_key]}），更新为：{current_value}")
            key_value_map[current_key] = current_value
            id_prompt_map[current_value] = current_key
            
            # 3.4 自增ID（确保全局唯一，跨CSV连续递增）
            global_id += 1
    
    # 4. 将Key-Value映射写入JSON文件（格式化输出，便于阅读）
    if not key_value_map:
        print("警告：未处理到有效数据，输出JSON文件将为空对象 {}")
    else:
        try:
            with open(output_json_path, "w", encoding="utf-8") as json_file:
                # indent=4：格式化缩进，sort_keys=True：按Key字母序排序（可选）
                json.dump(key_value_map, json_file, ensure_ascii=False, indent=4, sort_keys=True)
            print(f"处理完成！共生成 {len(key_value_map)} 组映射，已保存到：{output_json_path}")
        except Exception as e:
            print(f"写入JSON文件失败：{str(e)}")

    # 5. 生成ID-Prompt映射文件（可选）
    # print("正在生成ID-Prompt映射文件...")s
    if not id_prompt_map:
        print("警告：未生成ID-Prompt映射，跳过此步骤")
    else:
        print("正在生成ID-Prompt映射文件...")
        id_prompt_txt_path = os.path.splitext(output_json_path)[0] + "_id_prompt.json"
        try:
            with open(id_prompt_txt_path, "w", encoding="utf-8") as json_file:
                # indent=4：格式化缩进，sort_keys=True：按Key字母序排序（可选）
                json.dump(id_prompt_map, json_file, ensure_ascii=False, indent=4, sort_keys=True)
            print(f"处理完成！共生成 {len(id_prompt_map)} 组映射，已保存到：{id_prompt_txt_path}")
        except Exception as e:
            print(f"写入JSON文件失败：{str(e)}")
        print(f"ID-Prompt映射已保存到 {id_prompt_txt_path}")

def process_csv_files(csv_file_paths: List[str], output_txt_path: str) -> None:
    """
    处理多个CSV文件，生成Key-Value映射并写入TXT文件
    
    Args:
        csv_file_paths: CSV文件路径列表（如 ["data1.csv", "data2.csv"]）
        output_txt_path: 输出TXT文件的路径（如 "result_map.txt"）
    """
    # 1. 初始化全局自增ID（从1开始）和Key-Value字典
    global_id = 1
    key_value_map = {}
    id_prompt_map = {}
    
    # 2. 遍历每个CSV文件
    for csv_path in csv_file_paths:
        # 检查CSV文件是否存在
        if not os.path.exists(csv_path):
            print(f"警告：文件 {csv_path} 不存在，跳过处理")
            continue
        
        # 读取CSV文件（若某列缺失，pandas会报错，确保CSV包含指定列）
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"读取文件 {csv_path} 失败：{str(e)}，跳过处理")
            continue
        
        # 提取CSV文件名的前缀（以"_"分割后的第一个单词）
        csv_filename = os.path.basename(csv_path)  # 获取文件名（如 "data_2024.csv"）
        filename_prefix = csv_filename.split("_")[0]  # 分割后取第一个单词（如 "data"）
        
        # 3. 遍历CSV的每一行数据，生成Key和Value
        for idx, row in df.iterrows():
            # 3.1 生成Key：优先用synthetic_prompt，为空则用meta_prompt
            # 处理空值：pandas中NaN需用pd.isna判断，同时排除空字符串、纯空格
            synthetic_prompt = str(row.get("synthetic_prompt", "")).strip()
            meta_prompt = str(row.get("meta_prompt", "")).strip()
            
            if not pd.isna(synthetic_prompt) and synthetic_prompt != "":
                current_key = synthetic_prompt
            elif not pd.isna(meta_prompt) and meta_prompt != "":
                current_key = meta_prompt
            else:
                # 若两个字段都为空，跳过当前行并提示
                print(f"警告：CSV {csv_path} 第 {idx+1} 行的 synthetic_prompt 和 meta_prompt 均为空，跳过")
                continue
            
            # 3.2 生成Value：文件名前缀 + 全局自增ID
            current_value = f"{filename_prefix}_{global_id}"
            
            # 3.3 将Key-Value加入字典（若Key重复，后出现的会覆盖前一个，可根据需求调整）
            if current_key in key_value_map:
                print(f"注意：Key '{current_key}' 已存在（原Value：{key_value_map[current_key]}），将更新为新Value：{current_value}")
            key_value_map[current_key] = current_value
            id_prompt_map[current_value] = current_key
            
            # 3.4 自增ID+1
            global_id += 1
    
    # 4. 将Key-Value映射写入TXT文件
    if not key_value_map:
        print("警告：未处理到有效数据，TXT文件将为空")
    else:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            # 按Key排序写入（可选，若无需排序可删除sorted）
            for key, value in sorted(key_value_map.items(), key=lambda x: x[0]):
                f.write(f"{key}:{value}\n")
        print(f"处理完成！共生成 {len(key_value_map)} 组映射，已写入 {output_txt_path}")

    # 5. 生成ID-Prompt映射文件（可选）
    print("正在生成ID-Prompt映射文件...")
    if not id_prompt_map:
        print("警告：未生成ID-Prompt映射，跳过此步骤")
    else:
        print("正在生成ID-Prompt映射文件...")
        id_prompt_txt_path = os.path.splitext(output_txt_path)[0] + "_id_prompt.json"
        with open(id_prompt_txt_path, "w", encoding="utf-8") as f:
            for id_key, prompt in sorted(id_prompt_map.items(), key=lambda x: x[0]):
                f.write(f"{id_key}:{prompt}\n")
        print(f"ID-Prompt映射已保存到 {id_prompt_txt_path}")

# ------------------- 示例调用 -------------------
if __name__ == "__main__":
    # 1. 配置输入CSV文件路径（可添加多个，支持不同目录下的文件）
    csv_files = [
        "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/colors_composition_prompts.csv",          # 当前目录下的CSV
        "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/counting_prompts.csv",  # 其他目录下的CSV
        "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/size_compositions_prompts.csv",
        "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/spatial_compositions_prompts.csv"
    ]
    
    # # 2. 配置输出TXT文件路径
    # output_txt = "key_value_map.txt"
    
    # # 3. 执行处理
    # process_csv_files(csv_file_paths=csv_files, output_txt_path=output_txt)

    # 2. 配置输出JSON文件路径（建议放在便于查找的位置）
    output_json_file = "key_value_mapping.json"
    
    # 3. 执行处理流程
    process_csv_to_json(csv_file_paths=csv_files, output_json_path=output_json_file)