import json
import pickle
from pycocotools.coco import COCO

def collect_coco_info(text_inputs_path, captions_path, instances_path, output_pkl_path):
    """
    从文本输入文件中收集coco_val_id，然后从COCO标注中提取对应信息
    
    参数:
        text_inputs_path: tifa_v1.0_text_inputs.json文件路径
        captions_path: captions_val2014.json文件路径
        instances_path: instances_val2014.json文件路径
        output_pkl_path: 输出pickle文件路径
    """
    # 1. 从文本输入文件中收集所有coco_val_id
    print(f"读取文本输入文件: {text_inputs_path}")
    with open(text_inputs_path, 'r', encoding='utf-8') as f:
        text_inputs = json.load(f)
    
    # 提取并去重coco_val_id
    coco_val_ids = set()
    coco_val_id_map = {}
    for item in text_inputs:
        if 'coco_val_id' in item:
            coco_val_ids.add(item['coco_val_id'])
            coco_val_id_map[item['coco_val_id']] = item['caption']
    
    coco_val_ids = list(coco_val_ids)
    print(f"共收集到 {len(coco_val_ids)} 个唯一的coco_val_id")
    
    # 2. 初始化COCO API
    print(f"加载COCO标注文件...")
    coco_capt = COCO(captions_path)  # 用于获取caption
    coco_inst = COCO(instances_path)  # 用于获取layout信息
    
    # 3. 处理每个coco_val_id，提取信息
    result = []
    for idx, val_id in enumerate(coco_val_ids):
        # 转换为整数ID（COCO标注中的ID是整数）
        try:
            img_id = int(val_id)
        except ValueError:
            print(f"警告: 无效的coco_val_id {val_id}，跳过处理")
            continue
        
        # 检查图片是否存在
        if img_id not in coco_inst.imgs:
            print(f"警告: 图片ID {img_id} 不在实例标注中，跳过处理")
            continue
        
        # 获取图片信息
        img = coco_inst.loadImgs(img_id)[0]
        img_width, img_height = img['width'], img['height']
        
        # 获取caption（取第一个）
        capt_ids = coco_capt.getAnnIds(imgIds=img_id)
        if not capt_ids:
            print(f"警告: 图片ID {img_id} 没有对应的caption，跳过处理")
            continue
        
        captions = coco_capt.loadAnns(capt_ids)
        # prompt = captions[0]['caption'].strip().lower()
        prompt = captions[0]['caption'].strip()

        if prompt != coco_val_id_map.get(val_id, "").strip():
            # 同图不同标注 -- 这可能会导致在生成图片时出现错误
            print(f"警告: coco_val_id {val_id} 的caption与COCO标注不匹配")
            print(f"  文本输入文件中的caption: {coco_val_id_map.get(val_id, '').strip()}")
            print(f"  COCO标注中的caption: {prompt}")
            # tifa_v1.0_text_inputs.json：该文件可能是第三方数据集（如 Tifa-Bench）基于 COCO 图片重新标注或筛选的 caption，可能只选取了其中一个标注，或对原始标注进行了修改（如简化、润色）。

            prompt = coco_val_id_map.get(val_id, "").strip()
        
        # 构建结果字典
        item = {'prompt': prompt}
        
        # 获取实例标注（目标检测框）
        ann_ids = coco_inst.getAnnIds(imgIds=img_id)
        if ann_ids:
            anns = coco_inst.loadAnns(ann_ids)
            
            # 处理每个目标，最多取10个
            for i, ann in enumerate(anns[:10]):
                # 获取类别名称
                cat = coco_inst.loadCats(ann['category_id'])[0]
                description = cat['name'].strip()
                
                # 转换边界框格式 [x, y, width, height] -> [x1, y1, x2, y2]
                # 并归一化到512x512尺寸
                x, y, w, h = ann['bbox']
                x1 = int(x * 512 / img_width)
                y1 = int(y * 512 / img_height)
                x2 = int((x + w) * 512 / img_width)
                y2 = int((y + h) * 512 / img_height)
                
                # 确保坐标在有效范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(512, x2), min(512, y2)
                
                item[str(i)] = {
                    'description': description,
                    'mask': [x1, y1, x2, y2]
                }
        
        result.append(item)
        
        # 打印进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(coco_val_ids)} 个ID")
    
    # 4. 保存结果到pickle文件
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"处理完成，共提取 {len(result)} 条有效数据，已保存到 {output_pkl_path}")
    return result

if __name__ == "__main__":
    # 配置文件路径
    TEXT_INPUTS_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/tifa_v1.0_text_inputs.json"
    CAPTIONS_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/captions_val2014.json"
    INSTANCES_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/annotations/instances_val2014.json"
    OUTPUT_PKL_PATH = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/coco_caption_layout.pkl"  # 输出的pickle文件路径
    
    # 执行处理
    collect_coco_info(TEXT_INPUTS_PATH, CAPTIONS_PATH, INSTANCES_PATH, OUTPUT_PKL_PATH)
    