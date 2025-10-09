import os
import pickle
import json

def generate_mappings(input_dir, output_dir):
    """
    读取所有pkl文件，生成{image_id: caption}和{caption: image_id}映射表
    并分别保存为JSON文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化两个映射表
    image_id_to_caption = {}
    caption_to_image_id = {}
    
    # 获取所有pkl文件
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    print(f"找到 {len(pkl_files)} 个pkl文件，开始处理...")
    
    for filename in pkl_files:
        file_path = os.path.join(input_dir, filename)
        
        # 读取pkl文件
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
        
        # 处理每个条目
        for item in data_list:
            image_id = item['image_id']
            caption = item['caption']
            
            # 处理image_id到caption的映射
            # 如果一个image_id对应多个caption，只保留第一个
            if image_id not in image_id_to_caption:
                image_id_to_caption[image_id] = caption
            
            # 处理caption到image_id的映射
            # 如果一个caption对应多个image_id，只保留第一个
            if caption not in caption_to_image_id:
                caption_to_image_id[caption] = str(image_id)
    
    # 定义输出文件路径
    img_id_to_caption_path = os.path.join(output_dir, "image_id_to_caption.json")
    caption_to_img_id_path = os.path.join(output_dir, "caption_to_image_id.json")
    
    # 保存image_id到caption的映射（注意JSON的键必须是字符串，所以需要转换）
    with open(img_id_to_caption_path, 'w', encoding='utf-8') as f:
        # 将整数image_id转换为字符串以符合JSON规范
        json.dump({str(k): v for k, v in image_id_to_caption.items()}, f, ensure_ascii=False, indent=2)
    
    # 保存caption到image_id的映射
    with open(caption_to_img_id_path, 'w', encoding='utf-8') as f:
        json.dump(caption_to_image_id, f, ensure_ascii=False, indent=2)
    
    print(f"映射表生成完成：")
    print(f"- image_id到caption映射：{len(image_id_to_caption)} 条记录，保存至 {img_id_to_caption_path}")
    print(f"- caption到image_id映射：{len(caption_to_image_id)} 条记录，保存至 {caption_to_img_id_path}")
    
    return image_id_to_caption, caption_to_image_id

def main():
    # 配置文件路径
    input_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/semantic-object-accuracy-for-generative-text-to-image-synthesis/SOA/captions"
    output_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mappings"  # 输出目录
    
    # 生成映射表
    generate_mappings(input_dir, output_dir)

if __name__ == "__main__":
    main()
