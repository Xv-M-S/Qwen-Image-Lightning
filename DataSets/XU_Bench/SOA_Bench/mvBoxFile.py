import os
import shutil
import argparse

def copy_ground_truth(ground_truth_dir, target_root_dir):
    """
    将ground truth文件复制到对应的目标文件夹
    
    参数:
        ground_truth_dir: 存放ground truth pkl文件的目录
        target_root_dir: 包含所有label子文件夹的根目录
    """
    # 记录复制的文件数量
    copied_count = 0
    
    # 遍历所有ground truth文件
    for filename in os.listdir(ground_truth_dir):
        # 检查是否是pkl文件且文件名包含"ground_truth"
        if filename.endswith("_ground_truth.pkl") and os.path.isfile(os.path.join(ground_truth_dir, filename)):
            # 提取label文件夹名称（去除"_ground_truth.pkl"后缀）
            label_folder_name = filename.replace("_ground_truth.pkl", "")
            
            # 构建目标文件夹路径
            target_folder = os.path.join(target_root_dir, label_folder_name)
            
            # 检查目标文件夹是否存在
            if not os.path.exists(target_folder):
                print(f"警告: 目标文件夹不存在 - {target_folder}，跳过该文件")
                continue
            
            # 构建源文件和目标文件路径
            source_file = os.path.join(ground_truth_dir, filename)
            target_file = os.path.join(target_folder, filename)
            
            # 复制文件
            try:
                shutil.copy2(source_file, target_file)  # copy2会保留文件元数据
                copied_count += 1
                print(f"已复制: {filename} -> {target_folder}")
            except Exception as e:
                print(f"复制失败 {filename}: {str(e)}")
    
    print(f"\n操作完成，共复制了 {copied_count} 个文件")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将ground truth文件复制到对应的label文件夹")
    parser.add_argument("--ground_truth_dir", 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/ground_truth_results",
                      help="存放ground truth pkl文件的目录")
    parser.add_argument("--target_root_dir", 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGINAME",  # 根据实际目标根目录修改
                      help="包含所有label子文件夹的根目录")
    
    args = parser.parse_args()
    
    # 执行复制操作
    copy_ground_truth(args.ground_truth_dir, args.target_root_dir)
