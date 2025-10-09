import pickle

def load_pkl(file_path):
    """
    读取给定路径下的 .pkl 文件，并返回加载的对象。
    
    参数:
        file_path (str): .pkl 文件的路径。
        
    返回:
        unpickled_obj: 从 .pkl 文件中加载的对象。
    """
    try:
        with open(file_path, 'rb') as f:
            # 使用 pickle.load 来加载 .pkl 文件中的对象
            unpickled_obj = pickle.load(f)
            print("File loaded successfully.")
            return unpickled_obj
    except Exception as e:
        print("Failed to load file:", e)

if __name__ == "__main__":
    # 假设 .pkl 文件位于当前目录下，或者提供绝对路径
    pkl_file_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/eval_metrics/detection/UniDet-master/pklDir/spatialRes.pkl"
    data = load_pkl(pkl_file_path)
    
    # 根据需要处理数据
    if data is not None:
        # 示例：打印数据（根据实际存储的数据类型调整）
        print(data)