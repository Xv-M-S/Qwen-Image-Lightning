import os
import pickle
import shutil
import re
from tqdm import tqdm

# 配置路径
PKL_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res_ground_truth"  # PKL文件所在目录，可根据实际情况修改
SOURCE_IMAGE_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/extracted_images"
DEST_IMAGE_DIR = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_images"  # 新文件夹路径

# 图片文件名前缀（根据实际情况修改）
IMAGE_PREFIX = "COCO_val2014_"  # 假设前缀，可根据实际数据调整

def extract_image_id_from_filename(filename):
    """从图片文件名中提取image_id（如从'561256.png'中提取561256）"""
    # 使用正则表达式提取文件名中的数字部分
    match = re.match(r'^(\d+)\.\w+$', filename)
    if match:
        return int(match.group(1))
    return None

def get_image_ids_from_pkl(pkl_path):
    """从PKL文件中提取所有image_id（适配新格式）"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            data = [data]
            
        # 提取所有image_id（从字典的键中解析）
        image_ids = []
        for item in data:
            if isinstance(item, dict):
                # 遍历字典的键（图片文件名）
                for filename in item.keys():
                    img_id = extract_image_id_from_filename(filename)
                    if img_id is not None:
                        image_ids.append(img_id)
        
        # 去重
        return list(set(image_ids))
        
    except Exception as e:
        print(f"处理PKL文件 {pkl_path} 时出错: {str(e)}")
        return []

def copy_images_for_label(label_name, image_ids, source_root, dest_root):
    """为指定类别复制对应的图片"""
    # 创建目标文件夹
    label_name = label_name.replace("_ground_truth", "")
    dest_folder = os.path.join(dest_root, label_name)
    os.makedirs(dest_folder, exist_ok=True)
    
    # 源文件夹路径
    source_folder = os.path.join(source_root, label_name)
    if not os.path.exists(source_folder):
        print(f"警告: 源文件夹 {source_folder} 不存在，跳过该类别")
        return
    
    # 复制每个图片
    for image_id in image_ids:
        # 生成图片名称 {prefix}_{image_id:012d}.jpg
        image_name = f"{IMAGE_PREFIX}{image_id:012d}.jpg"
        source_path = os.path.join(source_folder, image_name)
        
        if os.path.exists(source_path):
            dest_path = os.path.join(dest_folder, image_name)
            # 避免重复复制
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
        else:
            print(f"警告: 图片 {source_path} 不存在，跳过")

def process_all_labels():
    """处理所有类别"""
    # 获取所有PKL文件
    pkl_files = [f for f in os.listdir(PKL_DIR) if f.endswith(".pkl")]
    
    if not pkl_files:
        print("未找到任何PKL文件")
        return
    
    # 创建目标根目录
    os.makedirs(DEST_IMAGE_DIR, exist_ok=True)
    
    # 处理每个PKL文件
    for pkl_file in tqdm(pkl_files, desc="处理类别"):
        # 提取类别名称（不含.pkl扩展名）
        label_name = os.path.splitext(pkl_file)[0]
        pkl_path = os.path.join(PKL_DIR, pkl_file)
        
        # 获取该PKL文件中的所有image_id
        image_ids = get_image_ids_from_pkl(pkl_path)
        print(f"{label_name} 的图片ID: {image_ids}")
        
        if image_ids:
            print(f"处理 {label_name}: 找到 {len(image_ids)} 个图片ID")
            # 复制对应的图片
            copy_images_for_label(label_name, image_ids, SOURCE_IMAGE_DIR, DEST_IMAGE_DIR)
        else:
            print(f"在 {pkl_file} 中未找到有效的image_id")
    
    print(f"所有图片已提取至 {DEST_IMAGE_DIR}")

if __name__ == "__main__":
    process_all_labels()
    