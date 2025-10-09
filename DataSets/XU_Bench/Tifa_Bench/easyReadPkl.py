import pickle
import sys

def read_pkl(file_path):
    """读取.pkl文件并返回内容"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except Exception as e:
        print(f"读取文件时出错：{str(e)}")
    return None

if __name__ == "__main__":
    data_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/coco_caption_layout.pkl"
    data = read_pkl(data_path)
    for i, item in enumerate(data):
        print(f"Item {i}:")
        print(item)
        print("-" * 20)
    print(f"总共读取 {len(data)} 条数据")