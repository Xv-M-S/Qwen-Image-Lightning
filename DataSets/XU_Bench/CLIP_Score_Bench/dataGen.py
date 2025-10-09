import os
import json
import shutil
from pathlib import Path
import logging

def setup_logging():
    """配置日志输出"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing.log'),
            logging.StreamHandler()
        ]
    )

def load_prompt_mapping(json_path):
    """加载JSON文件中的id到prompt映射"""
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict):
            raise ValueError("JSON文件格式错误，应使用key-value结构")
            
        logging.info(f"成功加载prompt映射，共包含 {len(data)} 条记录")
        return data
        
    except json.JSONDecodeError:
        logging.error(f"JSON文件解析错误: {json_path}")
        raise
    except Exception as e:
        logging.error(f"加载prompt映射失败: {str(e)}")
        raise

def process_images(image_dir, prompt_mapping, output_image_dir, output_text_dir):
    """处理图片重命名和生成对应文本文件"""
    # 支持的图片文件后缀
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    
    # 统计变量
    total_processed = 0
    total_skipped = 0
    missing_prompts = 0
    
    # 遍历图片目录
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        
        # 跳过目录，只处理文件
        if not os.path.isfile(file_path):
            continue
            
        # 检查是否为图片文件
        ext = Path(filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            logging.debug(f"跳过非图片文件: {filename}")
            total_skipped += 1
            continue
            
        # 提取文件名前缀并处理
        try:
            # 分割文件名（不含后缀）
            name_parts = Path(filename).stem.split('_')
            
            if len(name_parts) < 2:
                logging.warning(f"文件名格式不符合要求（至少需要两个'_'分割部分）: {filename}")
                total_skipped += 1
                continue
                
            # 取前两部分作为新名称
            new_prefix = '_'.join(name_parts[:2])
            new_filename = f"{new_prefix}{ext}"
            new_image_path = os.path.join(output_image_dir, new_filename)
            
            # 复制并重命名图片
            shutil.copy2(file_path, new_image_path)  # 保留元数据
            
            # 查找对应的prompt
            if new_prefix not in prompt_mapping:
                logging.warning(f"未找到对应的prompt: {new_prefix}")
                missing_prompts += 1
                total_skipped += 1
                continue
                
            # 生成文本文件
            prompt = prompt_mapping[new_prefix].strip()
            text_filename = f"{new_prefix}.txt"
            text_path = os.path.join(output_text_dir, text_filename)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
                
            total_processed += 1
            logging.debug(f"处理完成: {filename} → {new_filename} 和 {text_filename}")
            
        except Exception as e:
            logging.error(f"处理文件 {filename} 时出错: {str(e)}")
            total_skipped += 1
            continue
    
    # 输出统计信息
    logging.info(f"处理完成 - 成功: {total_processed} 个, 跳过: {total_skipped} 个, 缺失prompt: {missing_prompts} 个")

import argparse

def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='处理图像路径和JSON文件路径参数')
    
    # 添加三个参数，设置默认值
    parser.add_argument('--raw-image-dir', 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/colorResGIName",
                      help='原始图像目录的路径')
    
    parser.add_argument('--prompt-json-path', 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/data_evaluate_LLM/HRS/key_value_mapping_id_prompt.json",
                      help='提示词JSON文件的路径')
    
    parser.add_argument('--output-root', 
                      type=str, 
                      default="/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/CLIP_Score_Bench/format_data/color_dataset",
                      help='输出根目录的路径')
    
    # 解析参数
    args = parser.parse_args()
    return args

def main():
    # 解析命令行参数
    args = parse_args()
    """主函数"""
    # 配置路径 - 根据实际情况修改
    RAW_IMAGE_DIR = args.raw_image_dir
    PROMPT_JSON_PATH = args.prompt_json_path
    OUTPUT_ROOT = args.output_root
    
    # 创建输出目录
    output_image_dir = os.path.join(OUTPUT_ROOT, "image")
    output_text_dir = os.path.join(OUTPUT_ROOT, "text")
    
    try:
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_text_dir, exist_ok=True)
        logging.info(f"输出目录已创建: {OUTPUT_ROOT}")
    except Exception as e:
        logging.error(f"创建输出目录失败: {str(e)}")
        return
    
    # 加载prompt映射
    try:
        prompt_mapping = load_prompt_mapping(PROMPT_JSON_PATH)
    except Exception as e:
        logging.error("无法继续处理，程序退出")
        return
    
    # 处理图片和文本
    process_images(RAW_IMAGE_DIR, prompt_mapping, output_image_dir, output_text_dir)
    
    # 输出最终结果路径
    logging.info(f"处理结果:")
    logging.info(f"图片目录: {output_image_dir}")
    logging.info(f"文本目录: {output_text_dir}")

if __name__ == "__main__":
    setup_logging()
    logging.info("开始执行图片和文本处理程序")
    main()
    logging.info("程序执行完毕")
    