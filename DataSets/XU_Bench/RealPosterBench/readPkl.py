import pickle

def read_and_print_pkl(file_path):
    """读取pkl文件并打印其内容"""
    try:
        # 以二进制读取模式打开pkl文件
        with open(file_path, 'rb') as f:
            # 加载pkl文件内容
            data = pickle.load(f)
        
        # 打印文件内容
        print("pkl文件内容如下：")
        print("-" * 50)
        print(data)
        print("-" * 50)
        print(f"数据类型：{type(data)}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
    except pickle.UnpicklingError:
        print(f"错误：文件 '{file_path}' 不是有效的pkl文件")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")

if __name__ == "__main__":
    # 替换为你的pkl文件路径
    pkl_file_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/MixBench/mixed_dataset.p"  # 这里可以修改为实际的pkl文件路径
    read_and_print_pkl(pkl_file_path)