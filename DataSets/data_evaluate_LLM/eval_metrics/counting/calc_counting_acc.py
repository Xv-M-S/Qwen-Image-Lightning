import pandas as pd
import csv
import json
import pickle
from tqdm import tqdm
import sys

import os
def convert_csv_2_txt():
    meta_f = open("meta_emotions.txt", "w")
    synthetic_f = open("synthetic_emotions.txt", "w")
    data = pd.read_csv("/home/eslam/Downloads/synthetic_emotion_prompts.csv").to_dict('records')
    for sample in data:
        meta_f.write(sample['meta_prompt']+"\n")
        synthetic_f.write(sample['synthetic_prompt']+"\n")
    meta_f.close()
    synthetic_f.close()


def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    gt_obj = []
    # import pdb; pdb.set_trace()
    for i, sample in enumerate(gt_data):
        temp_dict = {sample['obj1']: sample['n1']}
        if sample['n2'] > 0:
            temp_dict[sample['obj2']] = sample['n2']
        temp_dict["prompt"] = sample['synthetic_prompt']
        gt_obj.append(temp_dict)
       
    return gt_obj

def load_multi_pred(list_pkl_path, iter_idx):
    result = {iter_idx:{}}
    # result.update({iter_idx:{}})
    for pkl_path in list_pkl_path:
        with open(pkl_path, 'rb') as f:
            pred_data = pickle.load(f)
        result[iter_idx].update(pred_data[iter_idx])
    result = result[iter_idx]
    pred_objs = {}
    for img_id, v in result.items():
        temp_dict = {}
        for obj_id, v2 in v.items():
            temp_lst = []
            for item in v2:
                temp_lst.append(item[-1].lower())
            temp_dict[obj_id] = temp_lst

        pred_objs[img_id] = temp_dict
    return pred_objs

def load_pred(pkl_pth):
    with open(pkl_pth, 'rb') as f:
        pred_data = pickle.load(f)
    # import pdb; pdb.set_trace()
    print(pred_data)
    # pred_data = pred_data[iter_idx]
    # Keep class info only. Discard box coordinates info:
    pred_objs = {}
    for img_id, v in pred_data.items():
        temp_dict = {}
        for obj_id, v2 in v.items():
            temp_lst = []
            for item in v2:
                print(item)
                temp_lst.append(item[-1].lower())
            temp_dict[obj_id] = temp_lst

        pred_objs[img_id] = temp_dict

    return pred_objs


def cal_acc(gt_objs, pred_objs, global_id_map):
    # Calculate the Acc:
    true_pos = 0
    false_pos = 0
    false_neg = 0
   
    for img_id, sample in enumerate(gt_objs):
        
        # 校验
        prompt = sample["prompt"]
        if prompt not in global_id_map: continue
        img_id = global_id_map[prompt]
        del sample["prompt"]
        if not img_id in pred_objs.keys(): continue
        
        for obj_name, gt_num in sample.items():
            pred_num = 0
            
            for k, pred_obj in pred_objs[img_id].items():
                if obj_name in pred_obj:
                    pred_num += 1
            true_pos += min(gt_num, pred_num)
            false_pos = false_pos + max((pred_num-gt_num), 0)
            false_neg = false_neg + max((gt_num-pred_num), 0)
    
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return [precision, recall]

def read_json_to_map(json_file_path: str) -> dict:
    """
    读取JSON文件，构建并返回Key-Value映射（Python字典）
    
    Args:
        json_file_path: JSON文件的路径（如 "key_value_mapping.json"）
    
    Returns:
        dict: 从JSON文件解析出的Key-Value映射；若文件不存在/解析失败，返回空字典
    """
    # 1. 检查JSON文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：JSON文件 {json_file_path} 不存在")
        return {}
    
    # 2. 读取并解析JSON文件
    try:
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            # json.load() 会直接将JSON格式字符串转为Python字典
            key_value_map = json.load(json_file)
        
        # 3. 验证解析结果是否为字典（确保JSON格式符合Key-Value映射）
        if not isinstance(key_value_map, dict):
            print(f"错误：JSON文件 {json_file_path} 内容不是合法的Key-Value映射（需为JSON对象）")
            return {}
        
        print(f"成功读取JSON文件！共加载 {len(key_value_map)} 组Key-Value映射")
        return key_value_map
    
    # 处理常见异常（如JSON格式错误、权限不足等）
    except json.JSONDecodeError as e:
        print(f"错误：JSON文件 {json_file_path} 格式非法，解析失败：{str(e)}")
        return {}
    except PermissionError:
        print(f"错误：无权限读取JSON文件 {json_file_path}")
        return {}
    except Exception as e:
        print(f"读取JSON文件时发生未知错误：{str(e)}")
        return {}

if __name__ == "__main__":
    # 读取全局id配置
    json_file_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/key_value_mapping.json"
    global_id_map = read_json_to_map(json_file_path)

    # 输入参数
    in_pkl = sys.argv[1]
    gt_csv = sys.argv[2]
    iter_num = int(sys.argv[3])  # e.g., 1
    
    # Load GT:
    gt_objs = load_gt(csv_pth=gt_csv)
    precisions, recalls, f1 = [], [], []
    precisions_per_level = {0: [], 1: [], 2: []}
    recalls_per_level = {0: [], 1: [], 2: []}
    f1_per_level = {0: [], 1: [], 2: []}

    
    # Load Prediction:
    pred_objs = load_pred(pkl_pth=in_pkl)

    # Calculate the counting Accuracy:
    precision, recall = cal_acc(gt_objs, pred_objs, global_id_map)

    precision *= 100
    recall *= 100
    f1 = (2*precision*recall)/(precision+recall)

    print("precision : ",  precision, "%")
    print("recall : ", recall, "%")
    print("F1 Score : ", f1, "%")
