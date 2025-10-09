import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import json

# 1. 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, data_dir, global_id_file = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/key_value_mapping.json"):
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
        self.global_id_map = CustomDataset.read_json_to_map(global_id_file)
        print(self.global_id_map)
    

    def __len__(self):
        """返回数据集大小"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """根据索引返回数据样本"""

        # 返回 tensor 格式
        annotations = self.annotations[idx]
        id = self.getId(annotations["prompt"])
        # print(id)
        if  id != "unknown_id":
            annotations["id"] = id
            return annotations
        return None
    

    def getId(Self, name):
        if name in Self.global_id_map:
            return Self.global_id_map[name]
        else:
            return "unknown_id"
    

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
    

if __name__ == '__main__':
    # 创建数据集实例
    dataset = CustomDataset('/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p')

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 迭代数据加载器
    for batch in dataloader:
        # 处理数据批次
        # print(batch)
        pass