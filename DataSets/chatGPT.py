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

clientQwen = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_list(text):
    text =  text.replace(' ','')
    text =  text.replace('\n','')
    text =  text.replace('\t','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    return tuple(result)

def generate_box_qwen(text):
    # 示例 prompt，定义输出格式
    example_prompt = (
        "I want you to act as a programmer. I will provide the description of an image, "
        "you should output the corresponding layout of this image. Each object in the image is one "
        "rectangle or square box in the layout and size of boxes should be as large as possible "
        "compared to the image size. The size of the image is 512 * 512. "
        "You should return each object and the corresponding coordinate of its boxes.\n"
        "the prompt :\"three cats in the field\", \n"
        "cat: (51, 82, 399, 279)\n"
        "cat: (288, 128, 472, 299)\n"
        "cat: (27, 355, 418, 494)\n"
        "the prompt: \"a cat on the left of a dog on the road\"\n"
        "cat: (63, 196, 223, 394)\n"
        "dog: (289, 131, 466, 360)\n"
        "the prompt: \"four balls in the room\"\n"
        "ball: (72, 81, 254, 243)\n"
        "ball: (316, 44, 483, 218)\n"
        "ball: (287, 295, 453, 462)\n"
        "ball: (50, 323, 196, 484)\n"
        "the prompt: \"A donut to the right of a toilet\"\n"
        "donut: (287, 140, 467, 335)\n"
        "toilet: (31, 97, 216, 286)\n"
        "the prompt: \"A cat sitting on the top of a car\"\n"
        "car: (94, 236, 414, 407)\n"
        "cat: (124, 139, 273, 252)\n"
        "the prompt: \"A dog underneath a tree\"\n"
        "dog: (133, 232, 308, 445)\n"
        "tree: (121, 29, 324, 258)\n"
        "the prompt: \"a small ball is put on the top of a box on the table. there is a red vase on the right of the box on the table\"\n"
        "small ball: (92, 30, 165, 134)\n"
        "box: (93, 132, 205, 324)\n"
        "red vase: (214, 164, 297, 301)\n"
        "table: (36, 259, 418, 463)\n"
    )

    example_prompt = (
        "I want you to act as a programmer. I will provide the description of an image, "
        "you should output the corresponding layout of this image. Each object in the image is one "
        "rectangle or square box in the layout and size of boxes should be as large as possible "
        "compared to the image size. The size of the image is 512 * 512. "
        "You should return each object and the corresponding coordinate of its boxes (x1,y1,x2,y2).where:"
        "- (x1, y1) is the top-left corner"
        "- (x2, y2) is the bottom-right corner\n"
        "the prompt :\"a sign written on it \"engrossing read.\"\", \n"
        "sign: (xx, xx, xxx, xxx)\n"
        "\"engrossing read.\":(xx, xx, xxx, xxx)\n"
        "the prompt :\"a real scene of physics laboratory with a sign written on it 'wine glass and car, two worlds apart.'\""
        "laboratory: (xx, xx, xxx, xxx)\n"
        "sign: (xx, xx, xxx, xxx)\n"
        "\"wine glass and car, two worlds apart.\":(xx, xx, xxx, xxx)\n"
        "car: (xx, xx, xxx, xxx)\n"
        "wine glass: (xx, xx, xxx, xxx)\n"

    )

    full_prompt = example_prompt + text

    # 使用千问模型调用
    completion = clientQwen.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": example_prompt},
            {"role": "user", "content": text},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    completed_text = json.loads(completion.model_dump_json())["choices"][0]["message"]["content"]
    print('Qwen Response:', completed_text)

    # 解析返回的文本为对象名和边界框
    lines = completed_text.strip().split('\n')
    name_objects = []
    boxes_of_object = []

    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        # match = re.match(r'^-\s*\*\*([a-zA-Z\s]+)\*\*\s*:\s*$$(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)$$', line)
        # if match:
        #     obj_name = match.group(1).strip() 
        #     box = (int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5)))
        #     name_objects.append(obj_name)
        #     boxes_of_object.append(box)
        #     continue

        try:
            obj_name, box_str = line.split(":", 1)
            # obj_name = re.sub(r'[-*`#\s]', '', obj_name)  # 移除 - ** 等符号,包括中间的空格
            obj_name = re.sub(r'[-*`#]', '', obj_name)
            obj_name = obj_name.strip()
            box = text_list(box_str)  # 假设 text_list 是你定义的解析函数
            name_objects.append(obj_name)
            boxes_of_object.append(box)
        except Exception as e:
            print(f"Error parsing line: {line}, error: {e}")
            continue

    return name_objects, boxes_of_object


# 初始化客户端（建议将 API key 放在环境变量中，如 OPENAI_API_KEY）
clientGPT = OpenAI(os.getenv("OPENAI_API_KEY"))  # 替换为你的实际 API Key，或使用环境变量

def generate_box_gpt(text):
    # 示例 prompt，定义输出格式
    example_prompt = (
        "I want you to act as a programmer. I will provide the description of an image, "
        "you should output the corresponding layout of this image. Each object in the image is one "
        "rectangle or square box in the layout and size of boxes should be as large as possible "
        "compared to the image size. The size of the image is 512 * 512. "
        "You should return each object and the corresponding coordinate of its boxes.\n"
        "the prompt :\"three cats in the field\", \n"
        "cat: (51, 82, 399, 279)\n"
        "cat: (288, 128, 472, 299)\n"
        "cat: (27, 355, 418, 494)\n"
        "the prompt: \"a cat on the left of a dog on the road\"\n"
        "cat: (63, 196, 223, 394)\n"
        "dog: (289, 131, 466, 360)\n"
        "the prompt: \"four balls in the room\"\n"
        "ball: (72, 81, 254, 243)\n"
        "ball: (316, 44, 483, 218)\n"
        "ball: (287, 295, 453, 462)\n"
        "ball: (50, 323, 196, 484)\n"
        "the prompt: \"A donut to the right of a toilet\"\n"
        "donut: (287, 140, 467, 335)\n"
        "toilet: (31, 97, 216, 286)\n"
        "the prompt: \"A cat sitting on the top of a car\"\n"
        "car: (94, 236, 414, 407)\n"
        "cat: (124, 139, 273, 252)\n"
        "the prompt: \"A dog underneath a tree\"\n"
        "dog: (133, 232, 308, 445)\n"
        "tree: (121, 29, 324, 258)\n"
        "the prompt: \"a small ball is put on the top of a box on the table. there is a red vase on the right of the box on the table\"\n"
        "small ball: (92, 30, 165, 134)\n"
        "box: (93, 132, 205, 324)\n"
        "red vase: (214, 164, 297, 301)\n"
        "table: (36, 259, 418, 463)\n"
    )
    full_prompt = example_prompt + text

    # 使用新版 OpenAI API 调用
    response = clientGPT.completions.create(
        model="gpt-3.5-turbo-instruct",  # 替代 text-davinci-003（已弃用）
        prompt=full_prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # 提取生成的文本
    completed_text = response.choices[0].text
    print('GPT Response:', completed_text)

    # 解析返回的文本为对象名和边界框
    lines = completed_text.strip().split('\n')
    name_objects = []
    boxes_of_object = []

    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        try:
            obj_name, box_str = line.split(":", 1)
            obj_name = obj_name.strip()
            box = text_list(box_str)  # 假设 text_list 是你定义的解析函数
            name_objects.append(obj_name)
            boxes_of_object.append(box)
        except Exception as e:
            print(f"Error parsing line: {line}, error: {e}")
            continue

    return name_objects, boxes_of_object

clientGPTFree = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # api_key=os.getenv("DASHSCOPE_API_KEY"),
    api_key = "sk-KfTO4U0MJZIKwp9dF9CbFeEc3cA3494a9e97604cCfEbC544",
    base_url="https://free.v36.cm/v1/",
)

def generate_box_gpt_free(text):
    # 示例 prompt，定义输出格式
    example_prompt = (
        "I want you to act as a programmer. I will provide the description of an image, "
        "you should output the corresponding layout of this image. Each object in the image is one "
        "rectangle or square box in the layout and size of boxes should be as large as possible "
        "compared to the image size. The size of the image is 512 * 512. "
        "You should return each object and the corresponding coordinate of its boxes.\n"
        "the prompt :\"three cats in the field\", \n"
        "cat: (51, 82, 399, 279)\n"
        "cat: (288, 128, 472, 299)\n"
        "cat: (27, 355, 418, 494)\n"
        "the prompt: \"a cat on the left of a dog on the road\"\n"
        "cat: (63, 196, 223, 394)\n"
        "dog: (289, 131, 466, 360)\n"
        "the prompt: \"four balls in the room\"\n"
        "ball: (72, 81, 254, 243)\n"
        "ball: (316, 44, 483, 218)\n"
        "ball: (287, 295, 453, 462)\n"
        "ball: (50, 323, 196, 484)\n"
        "the prompt: \"A donut to the right of a toilet\"\n"
        "donut: (287, 140, 467, 335)\n"
        "toilet: (31, 97, 216, 286)\n"
        "the prompt: \"A cat sitting on the top of a car\"\n"
        "car: (94, 236, 414, 407)\n"
        "cat: (124, 139, 273, 252)\n"
        "the prompt: \"A dog underneath a tree\"\n"
        "dog: (133, 232, 308, 445)\n"
        "tree: (121, 29, 324, 258)\n"
        "the prompt: \"a small ball is put on the top of a box on the table. there is a red vase on the right of the box on the table\"\n"
        "small ball: (92, 30, 165, 134)\n"
        "box: (93, 132, 205, 324)\n"
        "red vase: (214, 164, 297, 301)\n"
        "table: (36, 259, 418, 463)\n"
    )
    full_prompt = example_prompt + text

    # 使用新版 OpenAI API 调用
    response = clientGPTFree.completions.create(
        model="gpt-3.5-turbo-0125",  # 替代 text-davinci-003（已弃用）
        prompt=full_prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # 提取生成的文本
    print('GPT Response:',  response)
    completed_text = response.choices[0].text
    print('GPT Response:', completed_text)

    # 解析返回的文本为对象名和边界框
    lines = completed_text.strip().split('\n')
    name_objects = []
    boxes_of_object = []

    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        try:
            obj_name, box_str = line.split(":", 1)
            obj_name = obj_name.strip()
            box = text_list(box_str)  # 假设 text_list 是你定义的解析函数
            name_objects.append(obj_name)
            boxes_of_object.append(box)
        except Exception as e:
            print(f"Error parsing line: {line}, error: {e}")
            continue

    return name_objects, boxes_of_object





def read_example_prompts(file_path):
    
    with open(file_path, 'r') as f:
       data = f.read()
    return data




def load_json(path_file):
    with open(path_file) as f:
        data = json.load(f)
    return data

def draw_box_2(text, boxes,output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    
    draw = ImageDraw.Draw(image)
    for i, bbox in enumerate(boxes):
        for box in bbox:
            if i==0:
               
                draw.rectangle([(box[0] * 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='red', width=6)
                
            elif i==1:
                draw.rectangle([(box[0]* 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='green', width=6)
            else:
                draw.rectangle([(box[0]* 512, box[1]* 512),(box[2]* 512, box[3]* 512)], outline='blue', width=6)
    image.save(os.path.join(output_folder, img_name))

def text_list(text):
    text =  text.replace(' ','')
    text =  text.replace('\n','')
    text =  text.replace('\t','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    return tuple(result)
def read_csv(path_file, t):
    list_prompts = []
    with open(path_file,'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >0: 
                if  row[1] == t:
                    list_prompts.append(row)
    return list_prompts

def read_txt_label(file_path):
    labels = {}
    with open(file_path, 'r') as f:
        for x in f:
            x = x.replace(' \n', '')
            x = x.replace('\n', '')
            x = x.split(',')
            labels.update({x[0]: x[2]})
    return labels

def draw_box(text, boxes,output_folder, img_name):
    width, height = 512, 512
    image = Image.new('RGB', (width, height), 'gray')
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Roboto-LightItalic.ttf", size=20)
    for i, box in enumerate(boxes):
        t = text[i]
        draw.rectangle([(box[0], box[1]),(box[2], box[3])], outline=128, width=2)
        mean_box_x, mean_box_y = int((box[0] + box[2] )/ 2) + int((box[1] + box[3] )/ 2)
        draw.text((mean_box_x, mean_box_y), t, fill=200,font=font )
    image.save(os.path.join(output_folder, img_name))

def save_img(folder_name, img, prompt, iter_id, img_id):
    os.makedirs(folder_name, exist_ok=True)
    img_name = str(img_id) + '_' + str(iter_id) + '_' + prompt.replace(' ','_')+'.jpg'
    img.save(os.path.join(folder_name, img_name))

def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    meta = []
    syn_prompt = []

    for sample in gt_data:
        meta.append([sample['meta_prompt']])
        syn_prompt.append([sample['synthetic_prompt']])
    return meta, syn_prompt

def load_box(pickle_file):
    with open(pickle_file,'rb') as f:
        data = pickle.load(f)
    return data
def read_txt_hrs(filename):
    result = []
    with open(filename) as f: 
        for x in f:
            result.append([x.replace('\n','')])
    return result

def format_box(names, boxes):
    result_name = []
    resultboxes = []
    for i, name in enumerate(names):
        name = remove_numbers(name)
        result_name.append('a ' + name.replace('_',' '))
        if name == 'person': 
            boxes[i] = boxes[i]
        resultboxes.append([boxes[i]])
    return result_name, np.array(resultboxes)

def remove_numbers(text):
    result = ''.join([char for char in text if not char.isdigit()])
    return result
def process_box_phrase(names, bboxes):
    d = {}
    for i, phrase in enumerate(names):
        phrase = phrase.replace('_',' ')
        list_noun = phrase.split(' ')
        for n in list_noun:
            n = remove_numbers(n)
            if not n in d.keys():
                d.update({n:[np.array(bboxes[i])/512]})
            else:
                d[n].append(np.array(bboxes[i])/512)
    return d

def Pharse2idx_2(prompt, name_box):
    prompt = prompt.replace('.','')
    prompt = prompt.replace(',','')
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    bbox_to_self_att = []
    for obj in name_box.keys():
        obj_position = []
        in_prompt = False
        for word in obj.split(' '):
            if word in prompt_list:
                obj_first_index = prompt_list.index(word) + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'s' in prompt_list:
                obj_first_index = prompt_list.index(word+'s') + 1
                obj_position.append(obj_first_index)
                in_prompt = True
            elif word +'es' in prompt_list:
                obj_first_index = prompt_list.index(word+'es') + 1
                obj_position.append(obj_first_index)
                in_prompt = True 
        if in_prompt :
            bbox_to_self_att.append(np.array(name_box[obj]))
        
            object_positions.append(obj_position)

    return object_positions, bbox_to_self_att
