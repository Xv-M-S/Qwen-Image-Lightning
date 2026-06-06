import os

# 配置项
ROOT_DIR = "Generated_Layouts"  # 根目录
OUTPUT_FILE = "all_optimized_prompts.txt"  # 合并后的总文件名

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    # 遍历根目录下的所有子文件夹
    for folder_name in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # 定位当前文件夹下的 optimized_prompts.txt
        prompt_file = os.path.join(folder_path, "optimized_prompts.txt")
        if not os.path.exists(prompt_file):
            print(f"跳过：{folder_name} 中不存在 optimized_prompts.txt")
            continue
        
        # 写入来源文件夹标识
        # out_f.write(f"===== 来源：{folder_name} =====\n")
        
        # 读取并写入内容
        with open(prompt_file, "r", encoding="utf-8") as in_f:
            content = in_f.read()
            out_f.write(content)
            # out_f.write("\n\n")  # 块之间空行分隔

print(f"✅ 合并完成！结果已保存到 {OUTPUT_FILE}")