import os
import json

def generate_image_id_json(folder_path, output_json_path):
    # 定义图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # 存储图片ID和路径的字典
    image_dict = {}
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件扩展名
        ext = os.path.splitext(filename)[1].lower()
        
        # 检查是否为图片文件
        if ext in image_extensions:
            # 分割文件名（不含扩展名）
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            
            # 提取前两部分并拼接成图片ID
            if len(parts) >= 2:
                image_id = '_'.join(parts[:2])
                # 添加到字典中（值为图片文件名）
                image_dict[image_id] = filename
            else:
                print(f"警告：文件名 '{filename}' 格式不符合要求，无法提取ID")
    
    # 将字典写入JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(image_dict, f, indent=4, ensure_ascii=False)
    
    print(f"已生成JSON文件：{output_json_path}")
    print(f"共处理 {len(image_dict)} 张图片")

if __name__ == "__main__":
    # 图片文件夹路径
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/tifaResGINAME")
    args = parser.parse_args()
    
    image_folder = args.image_folder
    
    # 输出JSON文件路径（可根据需要修改）
    # output_json = "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/tifaResGINAME/image_id_mapping.json"
    output_json = os.path.join(image_folder, "image_id_mapping.json")
    
    # 生成JSON文件
    generate_image_id_json(image_folder, output_json)
