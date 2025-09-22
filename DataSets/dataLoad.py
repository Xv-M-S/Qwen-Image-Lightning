import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
import pickle

# 1. 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): 数据根目录
            annotation_file (str): 标注文件路径（如 CSV、TXT）
        """
        self.data_dir = data_dir
        with open(self.data_dir, 'rb') as f:
            raw_data = pickle.load(f)

        self.annotations = []
        if isinstance(raw_data, dict):
            for k,v in raw_data.items():
                self.annotations.append(v)
        elif isinstance(raw_data, list):
            self.annotations = raw_data
        
        # print(raw_data)
        # print(self.annotations)
    

    def __len__(self):
        """返回数据集大小"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """根据索引返回数据样本"""

        # 返回 tensor 格式
        return self.annotations[idx]
    

if __name__ == '__main__':
    # 创建数据集实例
    dataset = CustomDataset('/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p')

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 迭代数据加载器
    for batch in dataloader:
        # 处理数据批次
        print(batch)
        pass