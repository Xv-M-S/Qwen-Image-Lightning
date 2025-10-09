import pandas as pd
import csv
import json
import pickle
from tqdm import tqdm
import sys
import os

def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    gt_list = []
    for sample in gt_data:
        # Objects:
        objs = [sample['obj1'], sample['obj2']]
        for i in range(3, 5):
            if type(sample['obj' + str(i)]) is str:  # check if there is an object
                objs.append(sample['obj' + str(i)])

        # Relations:
        relations = [sample['rel1']]
        if type(sample['rel2']) is str:  # check if there is a second relation
            relations.append(sample['rel2'])

        prompt = sample['meta_prompt']

        gt_list.append({ "prompt":prompt, "objs": objs, "relations": relations})

    return gt_list


def load_pred(pkl_pth):
    with open(pkl_pth, 'rb') as f:
        pred_data = pickle.load(f)

    print(pred_data)
    pred_objs = {}
    for img_id, v in pred_data.items():
        temp_dict = {}

        for obj_id, v2 in v.items():
            item = v2[0]  # remove duplicate objects
            temp_dict[obj_id] = {"cls": item[-1]}
            # convert coordinates to float instead of str:
            cords = [float(cord) for cord in item[:4]]
            temp_dict[obj_id]["cords"] = cords  # (xmin, ymin, xmax, ymax)  # origin = top left

        pred_objs[img_id] = temp_dict

    return pred_objs


def _get_box_area(obj):
    """
        obj: coordinates of object 1 (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = obj
    w, h = xmax - xmin, ymax - ymin
    area = w * h
    return area


def _check_large(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    a1 = _get_box_area(obj_1)
    a2 = _get_box_area(obj_2)
    if a1 > a2:
        return True
    else:
        return False


def _check_small(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    a1 = _get_box_area(obj_1)
    a2 = _get_box_area(obj_2)
    if a1 < a2:
        return True
    else:
        return False


def _sort_pred_obj(pred_objs, gt_objs):
    """
    Sorting the predicted objects based on the GT objects.
    pred_objs: dict of pred objs. key --> obj_id. val --> cls and cords.
    gt_objs: list of gt cls names.
    """
    sorted_pred_objs = {}
    for key, pred_obj in pred_objs.items():
        if pred_obj['cls'] in gt_objs:
            sorted_pred_objs[gt_objs.index(pred_obj['cls'])] = pred_obj
    return sorted_pred_objs


def cal_acc(gt_objs, pred_objs, global_id_map):
    bigger_words = ["larger", "bigger"]
    smaller_words = ["smaller"]

    true_count = 0
    true_sum = 0

    # 计算不同level的准确率
    level_1_sum = 0
    level_1_count = 0
    level_2_sum = 0
    level_2_count = 0
    level_3_sum = 0
    level_3_count = 0

    for img_id, sample in enumerate(gt_objs):
        # 校验是否存在
        prompt = sample['prompt']
        if prompt not in global_id_map:
            continue
            
        img_id = global_id_map[prompt]
        if img_id not in pred_objs:
            continue

        # 统计不同level的数量
        true_sum += 1
        if len(sample['objs']) ==2:
            level_1_sum += 1
        elif len(sample['objs']) ==3:
            level_2_sum += 1
        elif len(sample['objs']) ==4:
            level_3_sum += 1
        
        
  
        pred_cls = [pred_objs[img_id][obj_id]['cls'] for obj_id in pred_objs[img_id].keys()]

        # Check whether the image contains the correct classes or not:
        miss_flag = False
        for obj_cls in sample['objs']:
            if obj_cls in pred_cls:
                continue
            else:
                miss_flag = True
                break
        if miss_flag:
            continue

        # Sorting the predicted objects based on the GT objects
        sorted_pred_objs = _sort_pred_obj(pred_objs[img_id], sample['objs'])

        # Determine the hardness level based on the number of objects:
        if len(sample['objs']) == 2:
            # Easy level:
            if sample['relations'][0] in bigger_words:
                if _check_large(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
                    level_1_count += 1
            elif sample['relations'][0] in smaller_words:
                if _check_small(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
                    level_1_count += 1
            else:
                raise Exception("Sorry, this relation is not handled !")

        elif len(sample['objs']) == 3:
            # Medium level:
            # Check first relation:
            if sample['relations'][0] in bigger_words:
                if not _check_large(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    continue
            elif sample['relations'][0] in smaller_words:
                if not _check_small(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    continue

            # Check second relation:
            if sample['relations'][1] in bigger_words:
                if _check_large(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                    true_count += 1
                    level_2_count += 1
            elif sample['relations'][1] in smaller_words:
                if _check_small(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                    true_count += 1
                    level_2_count += 1

        elif len(sample['objs']) == 4:
            # Hard level:
            # Check first relation:
            if sample['relations'][0] in bigger_words:
                if not (_check_large(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']) and
                        (_check_large(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']))):
                    continue
            elif sample['relations'][0] in smaller_words:
                if not (_check_small(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']) and
                        (_check_small(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']))):
                    continue

            # Check second relation:
            if sample['relations'][1] in bigger_words:
                if _check_large(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']):
                    true_count += 1
                    level_3_count += 1
            elif sample['relations'][1] in smaller_words:
                if _check_small(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']):
                    true_count += 1
                    level_3_count += 1

        else:
            raise Exception("Sorry, number of objects should be between 1-4")

    acc = 100 * (true_count / true_sum)
    level_1_acc = 100 * (level_1_count / level_1_sum) if level_1_sum > 0 else 0
    level_2_acc = 100 * (level_2_count / level_2_sum) if level_2_sum > 0 else 0
    level_3_acc = 100 * (level_3_count / level_3_sum) if level_3_sum > 0 else 0
    print("Level 1 total samples: ", level_1_sum, "; correct samples: ", level_1_count)
    print("Level 2 total samples: ", level_2_sum, "; correct samples: ", level_2_count)
    print("Level 3 total samples: ", level_3_sum, "; correct samples: ", level_3_count)
    ret_acc = {"acc": acc, "level_1_acc": level_1_acc, "level_2_acc": level_2_acc, "level_3_acc": level_3_acc}
    return ret_acc
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

    in_pkl = sys.argv[1]
    gt_csv = sys.argv[2]

    # Load GT:
    gt_data = load_gt(csv_pth=gt_csv)

    # Load Predictions:
    pred_data = load_pred(pkl_pth=in_pkl)

    # Calculate the counting Accuracy:
    acc = cal_acc(gt_data, pred_data, global_id_map)

    print("----------------------------")
    print("   Easy level Results   ")
    print("Accuracy: ", acc["level_1_acc"], "%")
    print("   Medium level Results   ")
    print("Accuracy: ", acc["level_2_acc"], "%")
    print("   Hard level Results   ")
    print("Accuracy: ", acc["level_3_acc"], "%")
    print("----------------------------")
    print("   Average level Results   ")
    print("Averaged Accuracy: ", acc["acc"], "%")
