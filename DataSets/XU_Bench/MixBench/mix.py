"""
分别从color.p,spatial.p,size.p,count.p,visual_text.p中加载数据，进行消融实验，
每个中随机选取50个数据进行测试。
"""
import sys
from pathlib import Path

# 获取当前文件的父级目录，再获取其父级目录（即项目根目录）
root_dir = str(Path(__file__).resolve().parents[2])
print(root_dir)
sys.path.append(root_dir)
from dataLoad import CustomDataset
import numpy as np
import pickle

paths = ["/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/color.p",
         "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/counting.p",
         "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/size.p",
         "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/spatial.p",
         "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Visual_Text_Bench/converted_dataset.p"]

mix_data = []
for path in paths:
    data = CustomDataset(path)
    print(len(data))
    for i in range(50):
        idx = np.random.randint(len(data))
        print(data[idx])
        mix_data.append(data[idx])

with open('mixed_dataset.p', 'wb') as f:
    pickle.dump(mix_data, f)