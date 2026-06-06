from datasets import load_dataset
import os
import json
from PIL import Image
import io  # 新增：用于处理字节流

# 加载数据集
dataset = load_dataset("PosterCraft/Poster100K")
print(dataset)

# 创建存储图片的文件夹
os.makedirs("raw_images", exist_ok=True)
os.makedirs("normalized_images", exist_ok=True)

# 选择训练集
train_dataset = dataset["train"]

# 抽取10个样本（随机抽取10个样本）
num_samples = 10
samples = train_dataset.shuffle(seed=42).select(range(num_samples))  # 随机抽取更具代表性

# 准备存储元数据的字典
metadata = {}

# 处理每个样本
for sample in samples:
    # 提取样本信息
    file_name = sample["file_name"]
    caption = sample["caption"]
    mask_regions = sample["mask_regions"]
    raw_image_bytes = sample["image"]  # 这里是字节流
    normalized_path = sample["normalized_path"]
    
    # 将字节流转换为PIL Image对象
    raw_image = Image.open(io.BytesIO(raw_image_bytes))
    
    # 保存原始图片
    raw_image_path = os.path.join("raw_images", file_name)
    raw_image.save(raw_image_path)
    
    # 处理normalized图片（假设也是字节流格式）
    # 注意：根据数据集实际情况，normalized图片可能需要从其他字段获取
    # 如果normalized_path对应的是另一个图片字节流，需要调整获取方式
    # 这里假设normalized图片也是同一个image字段（如果不是请修改）
    normalized_image = Image.open(io.BytesIO(raw_image_bytes))  # 字节流转Image
    normalized_image_path = os.path.join("normalized_images", file_name)
    normalized_image.save(normalized_image_path)
    
    # 存储元数据
    metadata[file_name] = {
        "caption": caption,
        "mask_regions": mask_regions
    }

# 保存元数据到JSON文件
with open("poster_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"已成功提取{num_samples}个样本：")
print(f"- 原始图片保存至：raw_images/")
print(f"- 标准化图片保存至：normalized_images/")
print(f"- 元数据保存至：poster_metadata.json")