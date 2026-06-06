import os
import shutil
import argparse

def move_generated_images(source_dir, new_folder_name="generated_images"):
    """
    移动所有名称包含"generated"的图片到新文件夹，并保持原有的label子文件夹结构
    
    参数:
        source_dir: 源文件夹路径（包含所有label子文件夹）
        new_folder_name: 新文件夹名称，默认是"generated_images"
    """
    # 创建新文件夹根目录
    new_root = os.path.join(source_dir, new_folder_name)
    os.makedirs(new_root, exist_ok=True)
    print(f"已创建新文件夹根目录: {new_root}")
    
    # 记录移动的文件数量
    total_moved = 0
    
    # 遍历源目录下的所有子文件夹（如label_00_person等）
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        
        # 只处理以label_开头的子文件夹
        if os.path.isdir(item_path) and item.startswith("label_"):
            # 在新根目录下创建对应的子文件夹
            new_subfolder = os.path.join(new_root, item)
            os.makedirs(new_subfolder, exist_ok=True)
            
            # 统计当前子文件夹中移动的文件数
            subfolder_moved = 0
            
            print(f"正在处理文件夹: {item} -> 对应新文件夹: {new_subfolder}")
            
            # 遍历子文件夹中的文件
            for filename in os.listdir(item_path):
                # 检查文件名是否包含"generated"且是图片文件
                if "generated" in filename.lower():
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                        # 源文件路径
                        src = os.path.join(item_path, filename)
                        # 目标文件路径
                        dst = os.path.join(new_subfolder, filename)
                        
                        # 处理可能的文件名重复
                        if os.path.exists(dst):
                            name, ext = os.path.splitext(filename)
                            counter = 1
                            while os.path.exists(os.path.join(new_subfolder, f"{name}_{counter}{ext}")):
                                counter += 1
                            dst = os.path.join(new_subfolder, f"{name}_{counter}{ext}")
                            print(f"  文件名重复，已重命名为: {f'{name}_{counter}{ext}'}")
                        
                        # 移动文件
                        shutil.move(src, dst)
                        subfolder_moved += 1
                        total_moved += 1
                        # print(f"  已移动: {filename}")
            
            if subfolder_moved > 0:
                print(f"  该文件夹共移动了 {subfolder_moved} 个文件\n")
            else:
                print(f"  该文件夹没有符合条件的文件\n")
    
    print(f"操作完成，总计移动了 {total_moved} 个图片文件")
    print(f"所有文件已按原结构保存至: {new_root}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="移动包含'generated'的图片到带对应子文件夹的新目录")
    parser.add_argument("--source_dir", 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGI",
                      help="源文件夹路径（包含所有label子文件夹）")
    parser.add_argument("--new_folder", 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/expData_v2/raw-qwen-image/soaResGINAME",
                      help="新文件夹的根目录名称")
    
    args = parser.parse_args()
    
    # 执行移动操作
    move_generated_images(args.source_dir, args.new_folder)
