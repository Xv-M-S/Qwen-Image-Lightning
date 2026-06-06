import os
import subprocess
import argparse
from datetime import datetime

def process_pkl_files_individually(pkl_dir, save_root):
    """
    逐一处理每个pkl文件，为每个文件创建同名子目录，调用batchRun.py生成图片
    
    参数:
        pkl_dir: .pkl文件所在的文件夹路径
        save_root: 图片保存的根目录
    """
    # 获取所有.pkl文件
    pkl_files = [
        os.path.join(pkl_dir, f) 
        for f in os.listdir(pkl_dir) 
        if f.endswith('.pkl') and os.path.isfile(os.path.join(pkl_dir, f))
    ]
    
    if not pkl_files:
        print(f"在 {pkl_dir} 中未找到任何.pkl文件")
        return
    
    print(f"发现 {len(pkl_files)} 个.pkl文件，将逐一处理")
    
    # 创建保存图片的根目录（如果不存在）
    os.makedirs(save_root, exist_ok=True)
    
    # 记录处理日志
    log_file = os.path.join(save_root, "processing_log.txt")
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"===== 处理开始于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.write(f"总文件数: {len(pkl_files)}\n\n")
    
    # 逐一处理每个文件
    for idx, pkl_file in enumerate(pkl_files, 1):
        # 获取pkl文件名（不含扩展名）
        pkl_filename = os.path.basename(pkl_file)
        pkl_name_without_ext = os.path.splitext(pkl_filename)[0]
        
        # 创建与pkl文件同名的保存目录
        save_dir = os.path.join(save_root, pkl_name_without_ext)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n处理第 {idx}/{len(pkl_files)} 个文件: {pkl_filename}")
        print(f"保存目录: {save_dir}")
        
        # 记录当前文件信息
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"文件 {idx}: {pkl_filename}\n")
            log.write(f"保存目录: {save_dir}\n")
        
        try:
            id_map_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/mappings/caption_to_image_id.json"
            # 构建命令
            cmd = [
                "python", "batchRun.py",
                "--data_dir", pkl_file,
                "--save_path", save_dir,
                "--global_id_map", id_map_path
            ]
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # 记录成功信息
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"成功处理: {pkl_filename}\n")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"处理 {pkl_filename} 失败: {str(e)}\n"
            error_msg += f"标准错误输出: {e.stderr}\n"
            print(error_msg)
            
            # 记录错误信息
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"错误: {error_msg}\n")
        
        except Exception as e:
            error_msg = f"处理 {pkl_filename} 发生未知错误: {str(e)}\n"
            print(error_msg)
            
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"错误: {error_msg}\n")
    
    # 完成处理
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"\n===== 处理结束于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n\n")
    
    print("\n所有文件处理完毕！")
    print(f"处理日志已保存至: {log_file}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="逐一处理pkl文件并调用batchRun.py生成图片")
    parser.add_argument("--pkl_dir", 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/SOA_Bench/data_handle_new/sampled_captions_512res_clear_layout",
                      help=".pkl文件所在的文件夹路径")
    parser.add_argument("--save_root", 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/SOA_generated_images",
                      help="生成图片的保存根目录")
    
    args = parser.parse_args()
    
    # 开始逐一处理
    process_pkl_files_individually(
        pkl_dir=args.pkl_dir,
        save_root=args.save_root
    )
