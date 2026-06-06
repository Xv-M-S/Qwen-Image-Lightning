import os
import re

# 根目录
ROOT_DIR = "Generated_Layouts"
OUTPUT_FILE = "total_pure_english_prompts.txt"

# 核心规则：只要不包含 中/日/韩 文字，就保留
# 不管标点、符号、引号、逗号、数字，全都允许
def is_pure_english(text):
    text = text.strip()
    if not text:
        return False
    
    # 匹配 中/日/韩 文字
    if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text):
        return False
    return True


with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for folder in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        txt_path = os.path.join(folder_path, "optimized_prompts.txt")
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if is_pure_english(line):
                out_f.write(line + "\n")

print(f"✅ 提取完成！已保存到 {OUTPUT_FILE}")