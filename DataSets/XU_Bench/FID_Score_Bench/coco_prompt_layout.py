import pickle
from pycocotools.coco import COCO
import os
import json

def generate_coco_prompt_layout(inst_json_path, capt_json_path, output_pkl_path):
    """
    读取COCO标注，生成描述和布局信息，并保存为pickle文件
    
    参数:
        inst_json_path: instances_val2014.json的路径
        capt_json_path: captions_val2014.json的路径
        output_pkl_path: 输出pickle文件路径（.p后缀）
    返回:
        包含所有图片信息的列表
    """
    # 初始化COCO API
    coco_inst = COCO(inst_json_path)
    coco_capt = COCO(capt_json_path)
    
    # 获取所有图片ID
    img_ids = coco_inst.getImgIds()
    global_id_map = {}
    result = []
    
    # for img_id in img_ids[:100]:  # 可根据需要调整处理的图片数量
    for img_id in img_ids[:4000]:  # 可根据需要调整处理的图片数量
        # 获取图片信息
        img = coco_inst.loadImgs(img_id)[0]
        img_width, img_height = img['width'], img['height']
        
        # 获取该图片的字幕（取第一个字幕作为prompt）
        capt_ids = coco_capt.getAnnIds(imgIds=img_id)
        captions = coco_capt.loadAnns(capt_ids)
        if not captions:
            continue  # 跳过无字幕的图片
        prompt = captions[0]['caption'].strip().lower()

        # 构建全局id
        print(f"Processing image ID: {img_id} with prompt: {prompt}")
        global_id_map[prompt] = str(img_id)
        
        # 获取该图片的实例标注（目标检测框）
        ann_ids = coco_inst.getAnnIds(imgIds=img_id)
        anns = coco_inst.loadAnns(ann_ids)
        
        # 构建布局信息字典
        layout = {'prompt': prompt}
        for i, ann in enumerate(anns[:5]):  # 每张图最多取5个目标
            # 获取类别名称
            cat = coco_inst.loadCats(ann['category_id'])[0]
            description = cat['name'].strip()
            
            # 转换边界框为[ x1, y1, x2, y2 ]并归一化到512x512
            x, y, w, h = ann['bbox']
            x1 = int(x * 512 / img_width)
            y1 = int(y * 512 / img_height)
            x2 = int((x + w) * 512 / img_width)
            y2 = int((y + h) * 512 / img_height)
            
            # 确保坐标在有效范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(512, x2), min(512, y2)
            
            layout[str(i)] = {
                'description': description,
                'mask': [x1, y1, x2, y2]
            }
        
        result.append(layout)
        if len(result) % 100 == 0:
            print(f"已处理 {len(result)} 张图片")
    
    # 保存为pickle文件（.p后缀）
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"结果已保存到 {output_pkl_path}，共 {len(result)} 条数据")

    # 保存全局唯一的map文件到.json文件中
    with open('global_id_map.json', 'w', encoding='utf-8') as f:
        json.dump(global_id_map, f, ensure_ascii=False, indent=2)   # 中文+缩进
    
    return result

if __name__ == "__main__":
    # 标注文件路径（请根据实际路径修改）
    instances_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_val2014.json"
    captions_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_val2014.json"
    
    # 生成并保存pickle文件（.p后缀）
    coco_results = generate_coco_prompt_layout(
        instances_path,
        captions_path,
        output_pkl_path="coco_prompt_layout.p"  # 输出为.p文件
    )
    