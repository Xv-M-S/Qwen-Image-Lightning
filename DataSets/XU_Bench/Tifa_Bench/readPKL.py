import pickle
import os
import argparse

def read_pkl_file(file_path, max_items=5, max_key_length=10):
    """
    读取pkl文件并展示内容
    
    参数:
        file_path: pkl文件路径
        max_items: 最多展示的条目数量（用于列表/字典）
        max_key_length: 展示字典时最多显示的键长度
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return None
    
    # 检查文件后缀
    if not file_path.endswith('.pkl'):
        print(f"警告: 文件不是.pkl后缀 - {file_path}")
    
    try:
        # 读取pkl文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"成功读取文件: {file_path}")
        print(f"数据类型: {type(data).__name__}")
        
        # 根据数据类型展示内容
        if isinstance(data, list):
            print(f"列表长度: {len(data)} 条数据")
            if len(data) > 0:
                print(f"前 {min(max_items, len(data))} 条数据示例:")
                for i, item in enumerate(data[:max_items]):
                    print(f"\n第 {i+1} 条:")
                    print_item(item, indent=4)
        
        elif isinstance(data, dict):
            print(f"字典包含: {len(data)} 个键值对")
            print(f"前 {min(max_items, len(data))} 个键值对示例:")
            for i, (key, value) in enumerate(list(data.items())[:max_items]):
                print(f"\n键 {i+1}: {key}")
                print(f"值 {i+1}:")
                print_item(value, indent=4)
        
        else:
            print("数据内容:")
            print_item(data)
        
        return data
    
    except pickle.UnpicklingError:
        print(f"错误: 无法解析pkl文件，文件可能损坏或不是有效的pickle格式")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
    
    return None

def print_item(item, indent=0):
    """递归打印数据项，处理嵌套结构"""
    indent_str = " " * indent
    
    if isinstance(item, dict):
        print(f"{indent_str}字典 ({len(item)} 个键)")
        for key, value in item.items():
            print(f"{indent_str} 键: {key}")
            print(f"{indent_str} 值:")
            print_item(value, indent + 4)
    
    elif isinstance(item, list):
        print(f"{indent_str}列表 ({len(item)} 个元素)")
        for i, elem in enumerate(item[:3]):  # 只显示前3个元素
            print(f"{indent_str} 元素 {i+1}:")
            print_item(elem, indent + 4)
        if len(item) > 3:
            print(f"{indent_str} 还有 {len(item) - 3} 个元素未显示...")
    
    elif isinstance(item, (int, float, str, bool, type(None))):
        # 对于字符串，过长时截断显示
        if isinstance(item, str) and len(item) > 100:
            print(f"{indent_str}{item[:100]}... (截断显示，原长度 {len(item)})")
        else:
            print(f"{indent_str}{item}")
    
    else:
        # 其他类型显示类型和基本信息
        print(f"{indent_str}类型: {type(item).__name__}")
        print(f"{indent_str}内容: {str(item)[:100]}...")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='读取并展示.pkl文件内容')
    parser.add_argument('file_path', help='pkl文件的路径')
    parser.add_argument('--max-items', type=int, default=5, 
                      help='最多展示的条目数量（默认5）')
    args = parser.parse_args()
    
    # 读取并展示pkl文件
    read_pkl_file(args.file_path, max_items=args.max_items)

if __name__ == "__main__":
    main()
    """
    run:
    python readPKL.py /home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/Tifa_Bench/tifa_v1.0/coco_caption_layout.pkl --max-items 3
    """