from transformers import AutoModelForCausalLM, AutoTokenizer
import re

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
        "you just need to return the layout of the image, please do not add any other information."
    )

def load_Qwen3():
    # 438G    models--Qwen--Qwen3-235B-A22B-Instruct-2507
    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507" # too big for me
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model

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

def Qwen3Call(text, tokenizer, model):
    # prepare the model input
    messages=[
            {"role": "system", "content": example_prompt},
            {"role": "user", "content": text},
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("content:", content)


    # 解析返回的文本为对象名和边界框
    lines = content.strip().split('\n')
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


if __name__ == "__main__":
    tokenizer, model = load_Qwen3()
    prompt = "请帮我生成一个包含两个物体的图片描述，一个是红色的苹果，另一个是黄色的香蕉。"
    Qwen3Call(prompt, tokenizer, model)