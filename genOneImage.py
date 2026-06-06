import torch
import math
import os
import random
from typing import List, Dict
import argparse
from torch.utils.data import DataLoader

# 核心依赖导入
from pipeline.pipeline_qwenimage_regional import RegionalQwenImagePipeline
from pipeline.QwenImageTransformerForRegional import RegionalQwenImageTransformer
from pipeline.attentionUtil import register_attention_control
from pipeline.attentionControl import AttentionStore
from util.tool import draw_masks_on_image, visualize_mask_pairs
from config.boxLossConfig import boxConfig
from diffusers import FlowMatchEulerDiscreteScheduler
from DataSets.dataLoad import CustomDataset

# 全局配置
ENABLE_PROFILER = False
DEFAULT_HW = (512, 512)  # 默认宽高
POSITIVE_MAGIC = {
    "en": "Ultra HD, 4K, cinematic composition.",
    "zh": "超清，4K，电影级构图"
}
NEGATIVE_PROMPT = " "

# ---------------------- 核心工具函数 ----------------------
def load_model():
    """加载Qwen-Image模型（带LoRA）"""
    model_name = "Qwen/Qwen-Image"
    lora_path = "Qwen-Image-Lightning/Qwen-Image-Lightning-4steps-V1.0.safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 加载Transformer
    model = RegionalQwenImageTransformer.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    )
    
    # 配置调度器
    scheduler = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    })

    # 构建pipeline
    pipe = RegionalQwenImagePipeline.from_pretrained(
        model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
    )
    pipe.load_lora_weights(lora_path)
    pipe = pipe.to(device)
    pipe.transformer.enable_gradient_checkpointing()

    # 注册注意力控制
    controller = AttentionStore()
    register_attention_control(pipe, controller=controller)

    return pipe, controller

def prepare_regional_control(height, width, regional_prompt_mask_pairs, save_path):
    """预处理区域控制参数"""
    from util.tool import get_child_boxes
    from util.textComposition import visualize_structured_boxes_with_text

    regional_prompt_mask_pairs = get_child_boxes(regional_prompt_mask_pairs, width, height)
    visualize_structured_boxes_with_text(width, height, regional_prompt_mask_pairs, output_path=save_path)

    regional_prompts, regional_masks, regional_boxes, regional_child_boxes = [], [], [], []
    background_mask = torch.ones((height, width))

    # 解析区域参数
    for region in regional_prompt_mask_pairs.values():
        desc, mask, child_mask = region['description'], region['mask'], region['child_boxes']
        regional_boxes.append(mask)
        
        # 构建mask张量
        x1, y1, x2, y2 = mask
        mask_tensor = torch.zeros((height, width))
        mask_tensor[y1:y2, x1:x2] = 1.0
        background_mask -= mask_tensor

        regional_prompts.append(desc)
        regional_masks.append(mask_tensor)
        regional_child_boxes.append(child_mask)

    # 添加背景区域
    whole_regional_mask = torch.ones((height, width)) - background_mask
    if background_mask.sum() > 0:
        regional_prompts.append("a photo")
        regional_masks.append(background_mask)

    return regional_prompts, regional_masks, regional_boxes, whole_regional_mask, regional_prompt_mask_pairs, regional_child_boxes

def vertify_input(batch):
    """输入数据校验与格式统一"""
    if isinstance(batch.get('prompt'), List):
        batch['prompt'] = batch['prompt'][0]
    if isinstance(batch.get('id'), List):
        batch['id'] = batch['id'][0]
    
    for k, v in batch.items():
        if k not in ['prompt', 'id']:
            if isinstance(v.get('description'), List):
                v['description'] = v['description'][0]
            if isinstance(v.get('mask', [])[0], torch.Tensor):
                v['mask'] = [t.item() for t in v['mask']]
    return batch

# ---------------------- 新增：便捷调用函数 ----------------------
def generate_image(
    prompt: str,
    save_path: str,
    regional_prompt_mask_pairs: Dict = None,
    height: int = 512,
    width: int = 512,
    seed: int = 68,
    mask_inject_steps: int = 0,
    double_inject_blocks_interval: int = 1,
    base_ratio: float = 0.1
):
    """
    便捷生成图片的函数
    :param prompt: 基础提示词
    :param save_path: 图片保存路径（含文件名，如 "output/coffee.png"）
    :param regional_prompt_mask_pairs: 区域控制的prompt-mask对，默认None
    :param height/width: 图片尺寸，默认512x512
    :param seed: 随机种子，默认68
    :param mask_inject_steps: 掩码注入步数，默认0
    :param double_inject_blocks_interval: 双注入块间隔，默认1
    :param base_ratio: 基础比例，默认0.1
    """
    # 加载模型
    pipe, controller = load_model()
    boxConfig.H, boxConfig.W = height, width

    # 处理区域控制
    regional_prompts, regional_masks, regional_boxes, whole_regional_mask, regional_child_boxes = [], [], [], None, []
    if regional_prompt_mask_pairs:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        regional_params = prepare_regional_control(
            height, width, regional_prompt_mask_pairs, 
            os.path.join(save_dir, "regional_layout.png")
        )
        regional_prompts, regional_masks, regional_boxes, whole_regional_mask, _, regional_child_boxes = regional_params

    # 选择最优种子（如果配置了迭代次数）
    min_loss = float('inf')
    if boxConfig.latents_choose_iter > 0:
        for _ in range(boxConfig.latents_choose_iter):
            random_seed = random.randint(0, 1000000)
            infer_loss = pipe.multi_step_loss(
                base_prompt=prompt + POSITIVE_MAGIC["en"],
                attention_store=controller,
                negative_prompt=NEGATIVE_PROMPT,
                width=width, height=height,
                num_inference_steps=4,
                true_cfg_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(random_seed),
                mask_inject_steps=mask_inject_steps,
                attention_kwargs={
                    "regional_prompts": regional_prompts,
                    "regional_masks": regional_masks,
                    "regional_boxes": regional_boxes,
                    "regional_child_boxes": regional_child_boxes,
                    "double_inject_blocks_interval": double_inject_blocks_interval,
                    "base_ratio": base_ratio,
                    "whole_regional_mask": whole_regional_mask,
                    "enable_whole_regional_mask": False
                },
                gaussian_smoothing_kwargs={"sigma": 0.5, "kernel_size": 3, "smooth_attentions": True}
            )
            if infer_loss < min_loss:
                seed, min_loss = random_seed, infer_loss

    # 生成图片
    image = pipe(
        base_prompt=prompt + POSITIVE_MAGIC["en"],
        attention_store=controller,
        negative_prompt=NEGATIVE_PROMPT,
        width=width, height=height,
        num_inference_steps=4,
        true_cfg_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        mask_inject_steps=mask_inject_steps,
        attention_kwargs={
            "regional_prompts": regional_prompts,
            "regional_masks": regional_masks,
            "regional_boxes": regional_boxes,
            "regional_child_boxes": regional_child_boxes,
            "double_inject_blocks_interval": double_inject_blocks_interval,
            "base_ratio": base_ratio,
            "whole_regional_mask": whole_regional_mask,
            "enable_whole_regional_mask": False
        },
        gaussian_smoothing_kwargs={"sigma": 0.5, "kernel_size": 3, "smooth_attentions": True}
    ).images[0]

    # 保存图片
    image.save(save_path)
    if regional_prompt_mask_pairs:
        draw_masks_on_image(save_path, regional_prompt_mask_pairs, output_path=f"{os.path.splitext(save_path)[0]}_layout.png")

    return image

# ---------------------- 批量处理函数（原run逻辑精简） ----------------------
def batch_generate(args):
    """批量生成图片（兼容原数据加载逻辑）"""
    # 配置超参数
    mask_inject_steps = 0
    double_inject_blocks_interval = 1
    base_ratio = 0.1

    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 加载数据
    dataload = CustomDataset(args.data_dir, args.global_id_map) if args.global_id_map else CustomDataset(args.data_dir)
    dataloader = DataLoader(dataload, batch_size=1, shuffle=False)

    # 加载模型
    pipe, controller = load_model()
    height, width = DEFAULT_HW
    boxConfig.H, boxConfig.W = height, width

    # 批量处理
    for idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        # 输入校验
        batch = vertify_input(batch)
        base_prompt = batch['prompt']
        save_name = batch['id']
        image_path = os.path.join(args.save_path, f"{save_name}_generated_image.png")

        # 跳过已生成的图片
        if os.path.exists(image_path):
            print(f"{save_name} 已生成，跳过")
            continue

        # 处理区域控制参数
        del batch['prompt'], batch['id']
        regional_params = prepare_regional_control(
            height, width, batch, 
            os.path.join(args.save_path, f"{idx}_regional.png")
        )
        regional_prompts, regional_masks, regional_boxes, whole_regional_mask, regional_prompt_mask_pairs, regional_child_boxes = regional_params

        # 可视化布局
        visualize_mask_pairs(regional_prompt_mask_pairs, width, height, os.path.join(args.save_path, f"{save_name}_visual_layout.png"))

        # 选择最优种子
        seed, min_loss = 68, float('inf')
        if boxConfig.latents_choose_iter > 0:
            for _ in range(boxConfig.latents_choose_iter):
                random_seed = random.randint(0, 1000000)
                infer_loss = pipe.multi_step_loss(
                    base_prompt=base_prompt + POSITIVE_MAGIC["en"],
                    attention_store=controller,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=width, height=height,
                    num_inference_steps=4,
                    true_cfg_scale=1.0,
                    generator=torch.Generator(device="cuda").manual_seed(random_seed),
                    mask_inject_steps=mask_inject_steps,
                    attention_kwargs={
                        "regional_prompts": regional_prompts,
                        "regional_masks": regional_masks,
                        "regional_boxes": regional_boxes,
                        "regional_child_boxes": regional_child_boxes,
                        "double_inject_blocks_interval": double_inject_blocks_interval,
                        "base_ratio": base_ratio,
                        "whole_regional_mask": whole_regional_mask,
                        "enable_whole_regional_mask": False
                    },
                    gaussian_smoothing_kwargs={"sigma": 0.5, "kernel_size": 3, "smooth_attentions": True}
                )
                if infer_loss < min_loss:
                    seed, min_loss = random_seed, infer_loss

        # 生成图片
        image = pipe(
            base_prompt=base_prompt + POSITIVE_MAGIC["en"],
            attention_store=controller,
            negative_prompt=NEGATIVE_PROMPT,
            width=width, height=height,
            num_inference_steps=4,
            true_cfg_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            mask_inject_steps=mask_inject_steps,
            attention_kwargs={
                "regional_prompts": regional_prompts,
                "regional_masks": regional_masks,
                "regional_boxes": regional_boxes,
                "regional_child_boxes": regional_child_boxes,
                "double_inject_blocks_interval": double_inject_blocks_interval,
                "base_ratio": base_ratio,
                "whole_regional_mask": whole_regional_mask,
                "enable_whole_regional_mask": False
            },
            gaussian_smoothing_kwargs={"sigma": 0.5, "kernel_size": 3, "smooth_attentions": True}
        ).images[0]

        # 保存图片
        image.save(image_path)
        draw_masks_on_image(image_path, regional_prompt_mask_pairs, output_path=os.path.join(args.save_path, f"{save_name}_layout_on_image.png"))

# ---------------------- 主函数 ----------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen-Image 区域控制图片生成")
    parser.add_argument('--save_path', type=str, required=True, help='保存路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--global_id_map', type=str, default=None, help='全局ID映射文件')
    args = parser.parse_args()

    # 性能分析（可选）
    if ENABLE_PROFILER:
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()

    # 批量生成
    batch_generate(args)

    if ENABLE_PROFILER:
        profiler.stop()
        profiler.print()

if __name__ == "__main__":
    # 方式1：命令行批量运行
    # main()

    # 方式2：直接调用generate_image函数（推荐）
    example_prompt = "A coffee shop entrance features a chalkboard sign reading '咖啡店' , and a neon light displaying '通义千问' ."
    generate_image(
        prompt=example_prompt,
        save_path="output/coffee_shop.png",
        # regional_prompt_mask_pairs=your_regional_pairs,  # 可选：区域控制参数
        height=512,
        width=512
    )