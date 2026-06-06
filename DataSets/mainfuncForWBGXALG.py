import os
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from openai import OpenAI
from WenBenGaiXieAndLayoutGen import rewrite_prompt_openai_style, prompt_to_layout, generate_image_scene, download_image
from visual_layout import visualize_layout
# ===================== 配置区 =====================
# 请确保已安装 openai 和 matplotlib: pip install openai matplotlib

# 初始化 OpenAI 兼容客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 请确保环境变量已设置
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ===================== 主处理流程 =====================
def process_prompts_from_file(input_txt_path, output_base_dir="output_projects",genImage = False):
    """
    读取文本文件，逐行处理 Prompt，生成布局并保存。
    
    :param input_txt_path: 输入的txt文件路径，每行一个Prompt
    :param output_base_dir: 输出的根目录
    """
    # 确保根输出目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 读取txt文件
    with open(input_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 处理每一行
    for idx, line in enumerate(lines):
        raw_prompt = line.strip()
        if not raw_prompt: # 跳过空行
            continue
            
        print(f"正在处理第 {idx+1} 行: {raw_prompt[:30]}...") # 打印前30个字符
        
        # 1. 优化 Prompt
        optimized_prompt = rewrite_prompt_openai_style(raw_prompt)
        if "调用出错" in optimized_prompt or "Error" in optimized_prompt:
            print(f"   Prompt 优化失败: {optimized_prompt}")
            continue
            
        # 2. 生成布局 JSON
        layout_data = prompt_to_layout(optimized_prompt)
        if isinstance(layout_data, str) and ("JSON解析错误" in layout_data or "调用出错" in layout_data):
            print(f"   布局生成失败: {layout_data}")
            continue
            
        # 3. 创建项目文件夹 (使用行号或清理后的Prompt作为文件夹名)
        # 清理文件名（移除非法字符）
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', raw_prompt[:50])
        project_dir = os.path.join(output_base_dir, f"{idx+1:03d}_{safe_name}")
        os.makedirs(project_dir, exist_ok=True)

        # --- 新增：定义优化后提示词的输出文件路径 ---
        optimized_file_path = os.path.join(project_dir, "optimized_prompts.txt")

        with open(optimized_file_path, 'w', encoding='utf-8') as opt_file:
            # 写入格式：原行号 | 优化后内容
            opt_file.write(f"{optimized_prompt}\n")
            # 强制刷新缓冲区，确保实时写入（可选）
            opt_file.flush()

        # 3. 生成图片
        if genImage:
            image_result = generate_image_scene(prompt_text=optimized_prompt)
            custom_file_name = "reference_image.png"
            print(f"生成结果: {image_result}")
            if image_result.startswith(('http://', 'https://')):
                save_path = download_image(
                    image_url=image_result,
                    save_path=project_dir,
                    file_name=custom_file_name
                )
                print(f"图片保存结果: {save_path}")
            else:
                print("未生成有效图片URL，无需下载")

        # 4. 保存 JSON 文件
        json_path = os.path.join(project_dir, "layout.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, ensure_ascii=False, indent=4)
            
        # 5. 可视化并保存图片
        img_path = os.path.join(project_dir, "preview.png")
        try:
            print(layout_data)
            # print(type(layout_data))
            print(type(layout_data[0]))
            # layout_data = *layout_data
            print(layout_data[0])
            # json_string = json.dumps(layout_data, ensure_ascii=False)
            visualize_layout(layout_data, save_path=img_path, show_plot=False)
            print(f"   成功生成: {project_dir}")
        except Exception as e:
            print(f"   可视化失败: {e}")

# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 假设你的提示词存储在 'prompts.txt' 文件中
    # input_file = "ch_prompts.txt" 
    # input_file = "en_prompt_glyphdraw2.txt"
    # input_file = "new_prompt.txt"
    # input_file = "collect_poster_prompt.txt"
    # input_file = "new_new_prompts.txt"
    input_file = "case_prompts.txt"

    if not os.path.exists(input_file):
        # 如果没有文件，创建一个示例文件
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("A modern minimalist composition with a light grey background. A slice of walnut cake with white icing on a white plate, with cutlery, a coffee cup, and a green plant.\n")
            # f.write("A dark textured baking sheet background with flour and an egg. Text: 10 WAYS, YOUR COOKING, to improve, skills.\n")
    
    # 开始处理
    process_prompts_from_file(input_file, output_base_dir="Generated_Layouts", genImage = False)