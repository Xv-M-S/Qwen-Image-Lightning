import cv2
import os
import os.path
import sys
import numpy as np
import pandas as pd
import sys
import pickle

import json
import os
def detect_color_hue_based(hue_value):
    if hue_value < 15:
        color = "red"
    elif hue_value < 22:
        color = "orange"
    elif hue_value < 39:
        color = "yellow"
    elif hue_value < 78:
        color = "green"
    elif hue_value < 131:
        color = "blue"
    else:
        color = "red"

    return color

coco_class_idx = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane":4, "bus":5, "train":6, "truck":7, "boat":8, "traffic light":9, "fire hydrant":10,
    "stop sign":11, "parking meter":12, "bench":13, "bird":14, "cat":15, "dog":16, "horse":17, "sheep":18, "cow":19, "elephant":20, "bear":21, "zebra":22,
    "giraffe":23, "backpack":24, "umbrella":25, "handbag":26, "tie":27, "suitcase":28, "frisbee":29, "skis":30, "snowboard":31, "sports ball":32,
    "kite":33, "baseball bat":34, "baseball glove":35, "skateboard":36, "surfboard":37, "tennis racket":38, "bottle":39, "wine glass":40, "cup":41, 
    "fork":42, "knife":43, "spoon":44, "bowl":45, "banana":46, "apple":47, "sandwich":48, "orange":49, "broccoli":50, "carrot":51, "hot dog":52, "pizza":53, 
    "donut":54, "cake":55, "chair":56, "couch":57, "potted plant":58, "bed":59, "dining table":60, "toilet":61, "tv":62, "laptop":63, "mouse":64, "remote":65, 
    "keyboard":66, "cell phone":67, "microwave":68, "oven":69, "toaster":70, "sink":71, "refrigerator":72, "book":73, "clock":74, "vase":75, "scissors":76, 
    "teddy bear":77, "hair drier":78, "toothbrush":79
}

def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    gt_list = []
    for sample in gt_data:
        # Objects:
        objs = [sample['obj1'], sample['obj2']]
        for i in range(3, 5):
            if type(sample['obj'+str(i)]) is str:  # check if there is an object
                objs.append(sample['obj'+str(i)])

        # Colors:
        colors = [sample['color1'], sample['color2']]
        for i in range(3, 5):
            if type(sample['color' + str(i)]) is str:  # check if there is other colors
                colors.append(sample['color' + str(i)])

        gt_list.append({"prompt": sample['meta_prompt'], "objs": objs, "colors": colors})

    return gt_list


def load_pred(pred_masks_names):
    img_masks_names_dict = {}
    for pred_masks_name in pred_masks_names:
        img_name = os.path.basename(pred_masks_name).split(".")[0]
        img_id = "_".join(img_name.split("_")[:2])
        if img_id not in img_masks_names_dict:
            img_masks_names_dict[img_id] = []
        img_masks_names_dict[img_id].append(pred_masks_name)
    return img_masks_names_dict


def cal_acc(gt_data, img_masks_names_dict, t2i_out_dir, in_masks_folder, global_id_map):
    true_counter = 0
    total_num_objs = 0

    t2i_map = {}
    for file in os.listdir(t2i_out_dir):
        if file.endswith('.png') or file.endswith('.jpg'):
            img_id = "_".join(file.split(".")[0].split("_")[:2])
            t2i_map[img_id] = file

    for idx, sample in enumerate(gt_data):  # loop on samples
        # 校验
        prompt = gt_data[idx]["prompt"]
        if prompt not in global_id_map:
            print("Warning: prompt not in global_id_map", prompt)
            continue
        img_id = global_id_map[prompt]
        if img_id not in t2i_map:
            print("Warning: img_id not in t2i_map", img_id)
            continue
        if img_id not in img_masks_names_dict:
            print("Warning: img_id not in img_masks_names_dict", img_id)
            continue
        
        
        gt_objs = gt_data[idx]["objs"]
        total_num_objs += len(gt_objs)

        gt_colors = gt_data[idx]["colors"]

        
        img_name = t2i_map[img_id]
        img = cv2.imread(os.path.join(t2i_out_dir, img_name))
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_frame = hsv_frame[:, :, 0]
        img_masks_names_per_sample = img_masks_names_dict[img_id]
        # import pdb; pdb.set_trace()
        for obj_idx in range(len(gt_objs)):  # loop on GT objs
            # 1) make sure the classes are correct:
            gt_obj_id = coco_class_idx[gt_objs[obj_idx]]
            img_masks_name_per_class = []
            for img_masks_name in img_masks_names_per_sample:
                if int(img_masks_name.split("_")[-1].split(".")[0]) == gt_obj_id:
                    img_masks_name_per_class.append(img_masks_name)
            if len(img_masks_name_per_class):
                # found some predictions match GT class
                # 2) make sure the color is correct:
                for i in range(len(img_masks_name_per_class)):
                    mask = cv2.imread(os.path.join(in_masks_folder, img_masks_name_per_class[i]), cv2.IMREAD_GRAYSCALE)
                    mask = mask / 255.0
                    mask = mask.astype(np.uint8)  # [0->1]
                    hsv_frame_masked = np.multiply(hsv_frame, mask)
                    avg_hue = hsv_frame_masked.sum() / np.count_nonzero(hsv_frame_masked)  # average hue component
                    detected_color = detect_color_hue_based(avg_hue)
                    if detected_color == gt_colors[obj_idx]:
                        true_counter += 1
                        break
    return 100*true_counter / total_num_objs


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
    """
    Example:
    python hue_based_color_classifier.py  'T2I_benchmark/data/colors/output/sd_v1' 'T2I_benchmark/data/colors/colors_composition_prompts.csv' 't2i_benchmark/data/t2i_out/sd_v1/colors'
    """
    # 读取全局id配置
    json_file_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/key_value_mapping.json"
    global_id_map = read_json_to_map(json_file_path)

    # Inputs:
    in_masks_folder = sys.argv[1]
    gt_csv = sys.argv[2]
    t2i_out_dir = sys.argv[3]

    # Load GT:
    gt_data = load_gt(csv_pth=gt_csv)
    pred_masks_names = os.listdir(in_masks_folder)

    # Load Predictions:
    img_masks_names_dict = load_pred(pred_masks_names)
    # Calculate the counting Accuracy:
    acc = cal_acc(gt_data, img_masks_names_dict, t2i_out_dir=t2i_out_dir, in_masks_folder=in_masks_folder, global_id_map=global_id_map)
    print("Accuracy:", acc)
