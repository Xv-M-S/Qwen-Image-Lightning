import os
# import openai
from openai import OpenAI  # 新版导入方式
import csv 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from urllib.request import urlopen
import pandas as pd
import pickle
import json 
import re
import dashscope

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
api_key = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def rewrite_prompt_openai_style(original_prompt):
    """
    使用类似 OpenAI 的风格调用 Qwen 模型优化提示词
    """
    # 构建系统指令
    system_instruction = """
    请将用户提供的原始提示词整合并改写为一段逻辑连贯、描述流畅的自然语言段落，重点强化画面的整体感与空间布局。
    
    **要求：**
    1. 在描述中必须明确加入物体之间的相对位置关系（如上下、左右、前后等）。
    2. 保持原始提示词的语言（英文保持英文，中文保持中文）。
    3. 仅返回优化后的提示词，不要包含解释。
    """

    try:
        response = client.chat.completions.create(
            model="qwen-max", # 指定模型
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": original_prompt}
            ],
            temperature=0.3 # 为了保持指令准确性，建议使用较低的温度
        )
        
        # 提取结果
        rewritten_text = response.choices[0].message.content.strip()
        return rewritten_text

    except Exception as e:
        return f"调用出错: {str(e)}"

# ==========================================
# 使用示例
# ==========================================

# if __name__ == "__main__":
#     # 测试用的原始提示词
#     raw_prompt = """
#     The background is a dark, textured baking sheet. 
#     There is scattered flour and a partially visible egg. 
#     Text: "10 WAYS", "YOUR COOKING", "to improve", "skills".
#     """
    
#     result = rewrite_prompt_openai_style(raw_prompt)
#     print("【优化后的提示词】")
#     print(result)



from openai import OpenAI
import json


def prompt_to_layout(scene_description):
    """
    将场景描述的自然语言提示词转换为带有归一化坐标的JSON布局数据。
    
    :param scene_description: str, 场景描述文本
    :return: dict or str, 解析后的JSON字典或错误信息
    """
    
    system_prompt = """请根据以下场景描述，生成一个JSON格式的布局数据。要求如下：
1. 坐标格式：所有边界框（box）必须使用归一化的 (cx, cy, w, h) 格式，即 [中心X, 中心Y, 宽度, 高度]，数值范围在0到1之间。
2. 命名规则：
   - 如果对象是文字（Text），name 字段必须直接使用该文字的具体内容（例如：'COOKING TURKEY'）。
   - 如果对象是图形元素，name 字段必须使用具体的实体名称（如 'Golden-Brown Pastry'），如果prompt中实体名称为中文，则使用中文，如果prompt中实体名称如果为英文，则使用英文，禁止使用 '图标'、'元素' 或 '图形' 等泛称。
3. 输出结构：仅输出标准的JSON代码，不要包含任何额外的解释或Markdown格式。
4. 例子：
输入：A dark textured baking sheet background with flour and an egg. Text: 10 WAYS, YOUR COOKING, to improve, skills.\
输出：[
    {
        "name": "Dark Textured Baking Sheet",
        "box": [
            0.5,
            0.5,
            1,
            1
        ]
    },
    {
        "name": "Flour Dusting",
        "box": [
            0.5,
            0.5,
            0.8,
            0.2
        ]
    },
    {
        "name": "Egg with Yolk Spilling Out",
        "box": [
            0.5,
            0.4,
            0.1,
            0.15
        ]
    },
    {
        "name": "10 WAYS TO IMPROVE YOUR COOKING SKILLS",
        "box": [
            0.5,
            0.6,
            0.8,
            0.1
        ]
    }
]






"""

    user_prompt = f"请处理以下场景描述：\n\n{scene_description}"
    
    try:
        response = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 # 低温度以获得更确定性的结构化输出
        )
        
        # 获取模型输出的文本
        raw_output = response.choices[0].message.content.strip()
        
        # 尝试去除可能存在的Markdown代码块标记（```json）
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:].strip()
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3].strip()
        
        # 解析JSON
        layout_data = json.loads(raw_output)
        return layout_data

    except json.JSONDecodeError as e:
        return f"JSON解析错误: {str(e)}\n模型输出内容:\n{raw_output}"
    except Exception as e:
        return f"调用出错: {str(e)}"

# ==========================================
# 使用示例
# ==========================================

# if __name__ == "__main__":
#     # 测试用的场景描述
#     test_description = """A solid dark brown background with a smooth texture serves as the backdrop. 
#     At the top, the text "tips for" is positioned above a baked item with a golden-brown crust resting on a white surface; a green leaf lies beside the pastry. 
#     Further down, the text "COOKING TURKEY" appears, with "15" located nearby. 
#     At the bottom, the text "SWIPE UP TO KNOW SECRETS TO A PERFECT JUICY TURKEY!" is displayed."""
    
#     result = prompt_to_layout(test_description)
    
#     # 打印格式化的JSON结果
#     print(json.dumps(result, indent=4, ensure_ascii=False))


import os
from http import HTTPStatus

def generate_image_scene(prompt_text: str, api_key: str = api_key) -> str:
    """
    生成图片，并修复 image_url 为 None 的问题
    """
    
    if api_key:
        dashscope.api_key = api_key

    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt_text}
            ]
        }
    ]

    try:
        response = dashscope.MultiModalConversation.call(
            model="qwen-image-2.0-pro",
            messages=messages,
            size='1024*1024'
        )
        
        if response.status_code == HTTPStatus.OK:
            # --- 关键修复点：检查返回内容的类型 ---
            # 模型可能返回图片，也可能返回文字（例如拒绝生成）
            output_content = response.output.choices[0].message.content
            print(output_content)
            
            for item in output_content:
                # 如果返回项中包含 'image'，才是真正的图片
                if 'image' in item:
                    image_url = item['image']
                    return image_url
            
            # 如果遍历完都没有找到 image，说明模型返回了纯文字
            # 我们返回模型的文字回复，方便调试
            text_response = "".join([item.get("text", "") for item in output_content])
            return f"Model returned text instead of image: {text_response}"
            
        else:
            return f"API Error: {response.code} - {response.message}"

    except Exception as e:
        return f"Exception: {str(e)}"

import requests

def download_image(image_url: str, save_path: str, file_name: str = None) -> str:
    """
    从URL下载图片并保存到指定路径
    :param image_url: 图片的URL地址
    :param save_path: 保存图片的文件夹路径（如：./images）
    :param file_name: 自定义文件名（可选，默认用URL中的文件名或随机命名）
    :return: 图片的完整保存路径（成功）/ 错误信息（失败）
    """
    # 校验URL是否有效（初步过滤非图片URL）
    if not image_url or not image_url.startswith(('http://', 'https://')):
        return f"无效的图片URL: {image_url}"

    # 创建保存目录（如果不存在）
    try:
        os.makedirs(save_path, exist_ok=True)
    except Exception as e:
        return f"创建保存目录失败: {str(e)}"

    # 处理文件名
    if not file_name:
        # 从URL提取文件名，若无则用时间戳命名
        file_name = image_url.split('/')[-1]
        # 过滤非法字符，避免文件保存失败
        file_name = file_name.replace('?', '').replace('&', '').replace('=', '')
        # 确保文件名有图片后缀
        if not any(file_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            file_name = f"{os.urandom(4).hex()}.png"  # 随机命名+png后缀

    # 拼接完整保存路径
    full_save_path = os.path.join(save_path, file_name)

    # 下载图片（设置超时和请求头，模拟浏览器请求）
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        # 流式下载，避免大图片占用过多内存
        response = requests.get(image_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()  # 抛出HTTP错误（如404、500）

        # 写入文件
        with open(full_save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return full_save_path  # 成功返回完整路径

    except requests.exceptions.RequestException as e:
        return f"下载图片失败: {str(e)}"
    except Exception as e:
        return f"保存图片失败: {str(e)}"

# # --- 调用示例 ---
# if __name__ == "__main__":
#     # 1. 配置参数
#     prompt = """冬日北京的都市街景，青灰瓦顶、朱红色外墙的两间相邻中式商铺比肩而立，
#     门前挂着红灯笼，地面有薄雪覆盖，行人穿着冬装，背景是灰色的胡同墙，
#     光影交织、动静相宜。"""
#     api_key = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量获取API Key
#     save_dir = "./generated_images"  # 图片保存目录
#     custom_file_name = "beijing_winter_scene.png"  # 自定义文件名

#     # 2. 生成图片URL
#     image_result = generate_image_scene(prompt_text=prompt, api_key=api_key)
#     print(f"生成结果: {image_result}")

#     # 3. 判断是否生成了有效URL，若是则下载
#     if image_result.startswith(('http://', 'https://')):
#         save_path = download_image(
#             image_url=image_result,
#             save_path=save_dir,
#             file_name=custom_file_name
#         )
#         print(f"图片保存结果: {save_path}")
#     else:
#         print("未生成有效图片URL，无需下载")


# --- 调用示例 ---
if __name__ == "__main__":
    # 定义提示词
    prompt = """冬日北京的都市街景，青灰瓦顶、朱红色外墙的两间相邻中式商铺比肩而立...（此处省略具体细节，保持与之前一致）...光影交织、动静相宜。"""
    
    # 调用函数
    result = generate_image_scene(prompt_text=prompt)
    save_dir = "./generated_images"
    custom_file_name = "beijing_winter_scene.png"  # 自定义文件名


      # 3. 判断是否生成了有效URL，若是则下载
    if result.startswith(('http://', 'https://')):
        save_path = download_image(
            image_url=result,
            save_path=save_dir,
            file_name=custom_file_name
        )
        print(f"图片保存结果: {save_path}")
    else:
        print("未生成有效图片URL，无需下载")

    print(result)



