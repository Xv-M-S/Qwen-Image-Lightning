import os
import requests
from tqdm import tqdm
import zipfile

def download_coco_dataset(save_dir, year='2017', split='train'):
    """
    下载COCO数据集的指定部分
    
    参数:
        save_dir: 保存数据集的目录
        year: 数据集年份（2014或2017）
        split: 数据集分割（train/val/test）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # COCO数据集URL模板
    base_url = f'http://images.cocodataset.org/zips/{split}{year}.zip'
    anno_url = f'http://images.cocodataset.org/annotations/annotations_trainval{year}.zip'
    
    # 下载图像压缩包
    print(f"开始下载{split}{year}图像...")
    img_zip_path = os.path.join(save_dir, f'{split}{year}.zip')
    download_file(base_url, img_zip_path)
    
    # 仅下载训练和验证集的标注（测试集无公开标注）
    if split in ['train', 'val']:
        print(f"开始下载{year}年标注文件...")
        anno_zip_path = os.path.join(save_dir, f'annotations_trainval{year}.zip')
        download_file(anno_url, anno_zip_path)
        
        # 解压标注文件
        print("解压标注文件...")
        with zipfile.ZipFile(anno_zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        os.remove(anno_zip_path)  # 删除压缩包
    
    # 解压图像文件
    print(f"解压{split}{year}图像...")
    img_dir = os.path.join(save_dir, f'{split}{year}')
    with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    os.remove(img_zip_path)  # 删除压缩包
    
    print(f"COCO {split}{year}数据集下载完成，保存至: {save_dir}")

def download_file(url, save_path):
    """带进度条的文件下载函数"""
    response = requests.get(url, stream=True, timeout=30)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

if __name__ == "__main__":
    # 配置下载参数
    COCO_SAVE_DIR = "./coco_dataset"  # 数据集保存路径
    COCO_YEAR = "2014"               # 可选2014或2017
    COCO_SPLIT = "val"             # 可选train/val/test
    
    # 下载数据集
    download_coco_dataset(
        save_dir=COCO_SAVE_DIR,
        year=COCO_YEAR,
        split=COCO_SPLIT
    )
    
    # 如需下载验证集，可再调用一次
    # download_coco_dataset(COCO_SAVE_DIR, year=COCO_YEAR, split='val')
    