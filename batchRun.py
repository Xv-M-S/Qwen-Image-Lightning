from pipeline.pipeline_qwenimage_regional import RegionalQwenImagePipeline, RegionalQwenImageAttnProcessor
from pipeline.QwenImageTransformerForRegional import RegionalQwenImageTransformer
from pipeline.attentionUtil import register_attention_control
from pipeline.attentionControl import AttentionStore
import torch
import math
from pyinstrument import Profiler
from util.tool import draw_masks_on_image, visualize_mask_pairs
import os
from config.boxLossConfig import boxConfig
from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
)
import random
from DataSets.dataLoad import CustomDataset
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

ENABLE_PROFILER = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_gb(model, dtype=torch.float32):
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4 if dtype == torch.float32 else 2
    total_bytes = total_params * bytes_per_param
    total_mb = total_bytes / (1024**2)
    total_gb = total_mb / 1024
    return total_gb

def load_model():
    model_name = "Qwen/Qwen-Image"
    lora_path = "Qwen-Image-Lightning/Qwen-Image-Lightning-4steps-V1.0.safetensors"

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"


    pipe_cls = RegionalQwenImagePipeline


    if lora_path is not None:
        model = RegionalQwenImageTransformer.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch_dtype)
        # assert os.path.exists(lora_path), f"Lora path {lora_path} does not exist"
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        pipe = pipe_cls.from_pretrained(
            model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
        )
        pipe.load_lora_weights(lora_path)
        controller = AttentionStore()
        register_attention_control(pipe, controller=controller)
    else:
        pipe = pipe_cls.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    # 启用梯度检查点 - 反向时重新计算中间值，节省内存空间
    pipe.transformer.enable_gradient_checkpointing()


    return pipe, controller

def get_hw():
    # Generate with different aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = 512, 512

    boxConfig.H = height
    boxConfig.W = width

    return height, width

def prepare_regional_control(height, width, regional_prompt_mask_pairs, save_path):
    ## regional prompt and mask settings
    from util.tool import get_child_boxes
    from util.textComposition import visualize_structured_boxes_with_text

    regional_prompt_mask_pairs = get_child_boxes(regional_prompt_mask_pairs, width, height)
    print(regional_prompt_mask_pairs)
    visualize_structured_boxes_with_text(width, height, regional_prompt_mask_pairs, output_path=save_path)

    regional_prompts = []
    regional_masks = []
    regional_boxes = []
    regional_child_boxes = []

    background_prompt = "a photo" # set by default, but if you want to enrich background, you can set it to a more descriptive prompt

    background_mask = torch.ones((height, width))
    for region_idx, region in regional_prompt_mask_pairs.items():
        description = region['description']
        mask = region['mask']
        child_mask = region['child_boxes']
        regional_boxes.append(mask)
        x1, y1, x2, y2 = mask
        mask = torch.zeros((height, width))
        mask[y1:y2, x1:x2] = 1.0
        background_mask -= mask
        regional_prompts.append(description)
        regional_masks.append(mask)
        regional_child_boxes.append(child_mask)

    # if regional masks don't cover the whole image, append background prompt and mask
    whole_regional_mask = torch.ones((height, width)) - background_mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    return regional_prompts, regional_masks, regional_boxes, whole_regional_mask, regional_prompt_mask_pairs, regional_child_boxes

def prepare_base_control():
    # base prompt settings
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup", and a neon light  displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".  '''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" , and a neon light  displaying "通义千问" . A poster showing a beautiful Chinese woman , and "π≈3.1415926-53589793-23846264-33832795-02384197" is written on the wall .'''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" , and a neon light  displaying "通义千问" . A poster showing a beautiful Chinese woman. the text "庐山云雾" is written on the wall .'''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup" , and a neon light  displaying "通义千问" . the text "庐山云雾" is written on the wall .'''
    base_prompt = '''the text "庐山云雾" is written on the wall .'''
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "咖啡店" , and a neon light  displaying "通义千问" . the text "π≈3.1415926" is written on the wall .'''
    # base_prompt = '''A coffee shop entrance, '''

    negative_prompt = " " # Recommended if you don't use a negative prompt.

    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
        "zh": "超清，4K，电影级构图" # for chinese prompt
    }

    return base_prompt, negative_prompt, positive_magic

def vertify_input(batch):
    if isinstance(batch['prompt'], List):
        batch['prompt'] = batch['prompt'][0]
    if isinstance(batch['id'], List):
        batch['id'] = batch['id'][0]
    for k,v in batch.items():
        if k != 'prompt' and k != "id" and isinstance(v['description'], List):
            batch[k]['description'] = v['description'][0]
        if k != 'prompt' and k != "id" and isinstance(v['mask'][0], torch.Tensor):
            batch[k]['mask'] = [t.item() for t in batch[k]['mask']]

    return batch

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Example script with save_path and data_dir arguments.")
    
    # 添加 save_path 参数
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to save the output files (required).'
    )
    
    # 添加 data_dir 参数
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing the input data (required).'
    )

    # 添加 global_id_map 参数
    parser.add_argument(
        '--global_id_map',
        type=str,
        default=None,
        help='Path to the global ID map JSON file (default: global_id_map.json).'
    )
    
    # 解析参数
    args = parser.parse_args()
    return args

def run():
    ## region control factor settings [超参数]
    mask_inject_steps = 0 # larger means stronger control, recommended between 5-10
    double_inject_blocks_interval = 1 # 1 means strongest control
    # single_inject_blocks_interval = 1 # 1 means strongest control
    base_ratio = 0.1 # smaller means stronger control
    # save_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/mixRes"
    args = parse_args()
    # save_path = "/home/sxm/flux-workspace/Qwen-Image-Lightning/expData/countingRes"
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## 数据加载
    # data_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/MixBench/mixed_dataset.p"
    # data_dir = "/home/sxm/flux-workspace/Qwen-Image-Lightning/DataSets/XU_Bench/LLM_Bench/counting.p"
    data_dir = args.data_dir
    if args.global_id_map is not None:
        dataload = CustomDataset(data_dir, args.global_id_map)
    else:
        dataload = CustomDataset(data_dir)
    dataloader = DataLoader(dataload, batch_size=1, shuffle=False)

    pipe, controller = load_model()
    height,width = get_hw()
    _, negative_prompt, positive_magic = prepare_base_control()

    index = 0
    for batch in dataloader:
        if batch is None:
            continue
        index = index + 1   
        batch = vertify_input(batch)
        base_prompt = batch['prompt']
        save_name = batch['id']

        # 判断是否已经运行过
        image_path = os.path.join(save_path, f"{save_name}_generated_image.png")
        if os.path.exists(image_path):
            print(f"{save_name} has been generated, skip")
            continue

        del batch['prompt']
        del batch['id']
        regional_prompts, regional_masks, regional_boxes, whole_regional_mask, regional_prompt_mask_pairs, regional_child_boxes = prepare_regional_control(height, width, batch, os.path.join(save_path, f"{index}_regional.png"))

        ## visual layout
        # visualize_mask_pairs(regional_prompt_mask_pairs, width, height, os.path.join(save_path, f"{index}_visual_layout.png"))
        visualize_mask_pairs(regional_prompt_mask_pairs, width, height, os.path.join(save_path, f"{save_name}_visual_layout.png"))

        seed = 68
        random_seed = 0
        min_loss = float('inf')
        if boxConfig.latents_choose_iter > 0:
            for _ in range(boxConfig.latents_choose_iter):
                random_seed = random.randint(0, 1000000)
                print(f"latents choose iter, random seed: {random_seed}, min loss: {min_loss}")
                # infer_loss = pipe.latents_choose(
                infer_loss = pipe.multi_step_loss(
                    base_prompt=base_prompt + positive_magic["en"],
                    attention_store=controller, # added for attention store
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=4,
                    true_cfg_scale=1.0, # do not use CFG
                    generator=torch.Generator(device="cuda").manual_seed(random_seed),
                    mask_inject_steps=mask_inject_steps, # inject mask
                    attention_kwargs={
                        "regional_prompts": regional_prompts,
                        "regional_masks": regional_masks,
                        "regional_boxes": regional_boxes,
                        "regional_child_boxes": regional_child_boxes,
                        "double_inject_blocks_interval": double_inject_blocks_interval,
                        # "single_inject_blocks_interval": single_inject_blocks_interval,
                        "base_ratio": base_ratio,
                        "whole_regional_mask": whole_regional_mask,  # 是否在相乘的时候只在mask上进行
                        "enable_whole_regional_mask": False
                    },
                    gaussian_smoothing_kwargs={
                        "sigma": 0.5,
                        "kernel_size": 3,
                        "smooth_attentions": True
                    }
                )
                
                if infer_loss < min_loss:
                    seed = random_seed
                    min_loss = infer_loss
        print(f"latents choose iter, final seed: {random_seed}, min loss: {min_loss}")
        image = pipe(
        # image = pipe.mutil_step_call(
            base_prompt=base_prompt + positive_magic["en"],
            attention_store=controller, # added for attention store
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=4,
            true_cfg_scale=1.0, # do not use CFG
            generator=torch.Generator(device="cuda").manual_seed(seed),
            mask_inject_steps=mask_inject_steps, # inject mask
            attention_kwargs={
                "regional_prompts": regional_prompts,
                "regional_masks": regional_masks,
                "regional_boxes": regional_boxes,
                "regional_child_boxes": regional_child_boxes,
                "double_inject_blocks_interval": double_inject_blocks_interval,
                # "single_inject_blocks_interval": single_inject_blocks_interval,
                "base_ratio": base_ratio,
                "whole_regional_mask": whole_regional_mask,  # 是否在相乘的时候只在mask上进行
                "enable_whole_regional_mask": False
            },
            gaussian_smoothing_kwargs={
                "sigma": 0.5,
                "kernel_size": 3,
                "smooth_attentions": True
            }
        ).images[0]

        # image_path = os.path.join(save_path, f"{index}_generated_image.png")
        image_path = os.path.join(save_path, f"{save_name}_generated_image.png")
        image.save(image_path)

        # visual layout on image
        # draw_masks_on_image(image_path, regional_prompt_mask_pairs, output_path=os.path.join(save_path, f"{index}_layout_on_image.png"))
        draw_masks_on_image(image_path, regional_prompt_mask_pairs, output_path=os.path.join(save_path, f"{save_name}_layout_on_image.png"))





if __name__ == "__main__":

    if ENABLE_PROFILER:
        profiler = Profiler()
        profiler.start()

    run()

    
    if ENABLE_PROFILER:
        profiler.stop()
        profiler.print()