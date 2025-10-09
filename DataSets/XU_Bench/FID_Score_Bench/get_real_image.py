import json
import os
import shutil

def copy_images_from_json(json_file_path):
    # 原始图片路径
    original_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/cocoapi/coco_dataset/raw/val2014"
    
    # 目标文件夹
    target_folder = "real_images"
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    
    # 读取JSON文件
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            
            # 提取所有ID
            ids = list(data.values())
            
            # 复制每个图片
            for img_id in ids:
                # 确保ID是12位数字，不足的前面补0
                formatted_id = f"{int(img_id):012d}"
                
                # 构建图片文件名
                img_filename = f"COCO_val2014_{formatted_id}.jpg"
                
                # 源文件路径
                src_path = os.path.join(original_path, img_filename)
                
                # 目标文件路径
                dest_path = os.path.join(target_folder, img_filename)
                
                # 检查源文件是否存在
                if os.path.exists(src_path):
                    # 复制文件
                    shutil.copy2(src_path, dest_path)
                    print(f"已复制: {img_filename}")
                else:
                    print(f"警告: 找不到文件 {src_path}")
                    
        print("所有图片处理完成")
        
    except FileNotFoundError:
        print(f"错误: 找不到JSON文件 {json_file_path}")
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {json_file_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    # 请将此处替换为你的实际JSON文件路径
    json_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/FID_Score_Bench/global_id_map.json"  # 假设JSON文件名为input.json并与脚本在同一目录
    copy_images_from_json(json_file)
