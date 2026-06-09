"""
paper_demo.py — 论文方法综合演示入口

基于论文《扩散模型结构与注意力机制分析》实现的三阶段布局控制方法：
1. NQE-EAA: 基于早期注意力对齐的噪声质量评估（潜在噪声筛选）
2. LALO: 布局感知注意力损失（推理优化）
3. Dynamic Masking: 动态软掩码策略（跨区域干扰抑制）

使用方法:
    python paper_demo.py

通过 boxConfig 中的开关控制各组件的启用/禁用:
    - boxConfig.use_nqe_eaa: 启用 NQE-EAA 噪声筛选
    - boxConfig.use_dynamic_mask: 启用动态软掩码
    - boxConfig.lossType: 选择 LALO 损失类型 ("diff", "rnb", "opt")
"""

import os
import random
import math
from typing import Tuple, Dict, List

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

# 本地模块导入
from pipeline.pipeline_qwenimage_regional import (
    RegionalQwenImagePipeline,
    RegionalQwenImageAttnProcessor
)
from pipeline.QwenImageTransformerForRegional import RegionalQwenImageTransformer
from pipeline.attentionUtil import register_attention_control
from pipeline.attentionControl import AttentionStore
from util.tool import (
    draw_masks_on_image,
    visualize_mask_pairs,
    get_child_boxes,
)
from util.textComposition import visualize_structured_boxes_with_text
from config.boxLossConfig import boxConfig
from testCase import regional_prompt_mask_pairs2 as regional_prompt_mask_pairs

# ===================== 全局配置 =====================
ENABLE_PROFILER = False
SAVE_BASE_PATH = "./paper_demo_output"
MODEL_NAME = "Qwen/Qwen-Image"
LORA_PATH = "Qwen-Image-Lightning/Qwen-Image-Lightning-4steps-V1.0.safetensors"


# ===================== 工具函数 =====================
def count_model_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size_gb(model: torch.nn.Module, dtype: torch.dtype = torch.float32) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4 if dtype == torch.float32 else 2
    total_gb = (total_params * bytes_per_param) / (1024 ** 3)
    return round(total_gb, 2)


# ===================== 参数配置 =====================
def configure_for_paper_demo():
    """
    配置 boxConfig 参数以启用论文方法。
    可根据需要调整各开关。
    """
    # === NQE-EAA: 潜在噪声筛选 ===
    boxConfig.use_nqe_eaa = True           # 启用 NQE-EAA
    boxConfig.nqe_eaa_candidates = 8       # 候选噪声数量
    boxConfig.nqe_eaa_alpha = 0.5          # 双向注意力平衡系数

    # === Dynamic Masking: 动态软掩码 ===
    boxConfig.use_dynamic_mask = True      # 启用动态软掩码
    boxConfig.dynamic_mask_gamma = 0.7     # txt→img 空间权重
    boxConfig.dynamic_mask_lambda = 0.7    # img→txt 区域指示权重
    boxConfig.dynamic_mask_delta = 0.3     # 跨区域文本注意力衰减
    boxConfig.dynamic_mask_mu = 1.5        # 图像区域内聚合权重
    boxConfig.dynamic_mask_eta = 0.3       # 图像局部平滑权重

    # === LALO: 布局感知注意力损失 ===
    boxConfig.lossType = "diff"            # "diff" = 论文中的 LALO
    boxConfig.switch_box_loss = True
    boxConfig.max_iter_to_alter = [0]      # 在第 0 步进行潜变量优化
    boxConfig.use_global_box_loss = True   # 使用均值-方差梯度归一化
    boxConfig.scale_factor = 0.1           # 布局优化学习率 η_layout

    # === 通用生成参数 ===
    boxConfig.latents_choose_iter = 0      # 使用 NQE-EAA 替代原有种子选择

    print("=" * 60)
    print("论文方法配置:")
    print(f"  NQE-EAA: {'启用' if boxConfig.use_nqe_eaa else '禁用'}")
    print(f"  候选噪声数: {boxConfig.nqe_eaa_candidates}")
    print(f"  动态软掩码: {'启用' if boxConfig.use_dynamic_mask else '禁用'}")
    print(f"  LALO 损失类型: {boxConfig.lossType}")
    print(f"  梯度归一化: {'均值-方差' if boxConfig.use_global_box_loss else '逐box平均'}")
    print("=" * 60)


# ===================== 模型加载 =====================
def load_qwen_image_model() -> Tuple[RegionalQwenImagePipeline, AttentionStore]:
    """加载 Qwen-Image 模型及 LORA 权重"""
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
        print(f"使用 GPU 加速，数据类型: {torch_dtype}")
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print("GPU 不可用，使用 CPU")

    controller = AttentionStore()

    if LORA_PATH is not None:
        transformer = RegionalQwenImageTransformer.from_pretrained(
            MODEL_NAME,
            subfolder="transformer",
            torch_dtype=torch_dtype
        )

        scheduler_config = {
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
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        pipe = RegionalQwenImagePipeline.from_pretrained(
            MODEL_NAME,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch_dtype
        )
        pipe.load_lora_weights(LORA_PATH)

        register_attention_control(pipe, controller=controller)
    else:
        pipe = RegionalQwenImagePipeline.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)

    pipe = pipe.to(device)
    pipe.transformer.enable_gradient_checkpointing()

    print(f"模型加载完成，可训练参数: {count_model_parameters(pipe.transformer):,}")
    print(f"模型显存占用: {calculate_model_size_gb(pipe.transformer, torch_dtype)} GB")

    return pipe, controller


# ===================== 区域参数准备 =====================
def get_image_resolution() -> Tuple[int, int]:
    aspect_ratio_config = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    width, height = aspect_ratio_config["16:9"]
    boxConfig.H = height
    boxConfig.W = width
    print(f"图片分辨率: {width}x{height} (16:9)")
    return height, width


def prepare_regional_control_params(height: int, width: int):
    regional_prompt_mask_pairs_processed = get_child_boxes(regional_prompt_mask_pairs, width, height)
    print(f"区域配置预处理完成，共 {len(regional_prompt_mask_pairs_processed)} 个区域")

    visualize_structured_boxes_with_text(
        width, height,
        regional_prompt_mask_pairs_processed,
        os.path.join(SAVE_BASE_PATH, "output_structured_boxes_with_text.png")
    )

    regional_prompts = []
    regional_masks = []
    regional_boxes = []
    regional_child_boxes = []
    background_prompt = "A coffee shop entrance, "
    background_mask = torch.ones((height, width))

    for region_idx, region in regional_prompt_mask_pairs_processed.items():
        region_desc = region['description']
        region_box = region['mask']
        child_boxes = region['child_boxes']

        x1, y1, x2, y2 = region_box
        region_mask = torch.zeros((height, width))
        region_mask[y1:y2, x1:x2] = 1.0

        background_mask -= region_mask

        regional_prompts.append(region_desc)
        regional_masks.append(region_mask)
        regional_boxes.append(region_box)
        regional_child_boxes.append(child_boxes)

    whole_regional_mask = torch.ones((height, width)) - background_mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)
        print("添加背景区域提示词")

    return (
        regional_prompts,
        regional_masks,
        regional_boxes,
        whole_regional_mask,
        regional_prompt_mask_pairs_processed,
        regional_child_boxes
    )


def prepare_base_prompt_params() -> Tuple[str, str, Dict[str, str]]:
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "咖啡店" ,
    and a neon light displaying "通义千问" . the text "π≈3.1415926" is written on the wall .'''
    negative_prompt = " "
    quality_magic_words = {
        "en": "Ultra HD, 4K, cinematic composition.",
        "zh": "超清，4K，电影级构图"
    }
    return base_prompt, negative_prompt, quality_magic_words


# ===================== 核心生成逻辑 =====================
def run_paper_demo():
    """执行论文方法的完整生成流程"""
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)

    # 1. 配置论文方法参数
    configure_for_paper_demo()

    # 2. 超参数
    hyper_params = {
        "mask_inject_steps": 5,
        "double_inject_blocks_interval": 1,
        "base_ratio": 0.1,
        "num_inference_steps": 4,
        "true_cfg_scale": 1.0
    }

    # 3. 加载模型
    pipe, attention_controller = load_qwen_image_model()
    img_height, img_width = get_image_resolution()
    base_prompt, negative_prompt, quality_magic = prepare_base_prompt_params()
    (
        regional_prompts,
        regional_masks,
        regional_boxes,
        whole_regional_mask,
        regional_prompt_mask_pairs,
        regional_child_boxes
    ) = prepare_regional_control_params(img_height, img_width)

    # 4. 可视化区域布局
    visualize_mask_pairs(
        regional_prompt_mask_pairs,
        img_width, img_height,
        os.path.join(SAVE_BASE_PATH, "visual_layout.png")
    )

    # 5. 噪声筛选（NQE-EAA 或原有种子选择）
    final_seed = 68
    attention_kwargs = {
        "regional_prompts": regional_prompts,
        "regional_masks": regional_masks,
        "regional_boxes": regional_boxes,
        "regional_child_boxes": regional_child_boxes,
        "double_inject_blocks_interval": hyper_params["double_inject_blocks_interval"],
        "base_ratio": hyper_params["base_ratio"],
        "whole_regional_mask": whole_regional_mask,
        "enable_whole_regional_mask": False
    }

    if boxConfig.use_nqe_eaa:
        print(f"\n=== NQE-EAA 噪声筛选 ===")
        print(f"生成 {boxConfig.nqe_eaa_candidates} 个候选噪声...")

        # Generate candidate noises
        candidates = []
        seeds = []
        for i in range(boxConfig.nqe_eaa_candidates):
            seed = random.randint(0, 1000000)
            seeds.append(seed)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn(
                (1, pipe.transformer.config.in_channels // 4,
                 img_height // pipe.vae_scale_factor // 2,
                 img_width // pipe.vae_scale_factor // 2),
                generator=gen, device="cuda", dtype=torch.bfloat16
            )
            candidates.append(noise)

        # 为 NQE-EAA 准备简化版 attention_kwargs（去除 double_inject_blocks_interval）
        nqe_kwargs = {
            "regional_prompts": regional_prompts,
            "regional_masks": regional_masks,
            "regional_boxes": regional_boxes,
            "regional_child_boxes": regional_child_boxes,
            "base_ratio": hyper_params["base_ratio"],
            "whole_regional_mask": whole_regional_mask,
            "enable_whole_regional_mask": False,
            "double_inject_blocks_interval": 1,
        }

        # 运行 NQE-EAA 评估
        from pipeline.noise_filtering import select_best_noise
        best_noise, best_idx, best_score = select_best_noise(
            pipe=pipe,
            candidate_noises=candidates,
            base_prompt=base_prompt + quality_magic["en"],
            bboxes=regional_boxes,
            height=img_height,
            width=img_width,
            attention_store=attention_controller,
            alpha=boxConfig.nqe_eaa_alpha,
            guidance_scale=hyper_params["true_cfg_scale"],
            num_inference_steps=hyper_params["num_inference_steps"],
            mask_inject_steps=0,
            attention_kwargs=nqe_kwargs,
        )
        final_seed = seeds[best_idx]
        print(f"NQE-EAA 最佳种子: {final_seed} (index={best_idx}, score={best_score:.4f})")
    elif boxConfig.latents_choose_iter > 0:
        # 原有种子选择方法
        print(f"\n=== 原有种子优化 (迭代{boxConfig.latents_choose_iter}次) ===")
        min_loss = float('inf')
        for iter_idx in range(boxConfig.latents_choose_iter):
            random_seed = random.randint(0, 1000000)
            current_loss = pipe.multi_step_loss(
                base_prompt=base_prompt + quality_magic["en"],
                attention_store=attention_controller,
                negative_prompt=negative_prompt,
                width=img_width, height=img_height,
                num_inference_steps=hyper_params["num_inference_steps"],
                true_cfg_scale=hyper_params["true_cfg_scale"],
                generator=torch.Generator(device="cuda").manual_seed(random_seed),
                mask_inject_steps=hyper_params["mask_inject_steps"],
                attention_kwargs=attention_kwargs,
                gaussian_smoothing_kwargs={
                    "sigma": 0.5, "kernel_size": 3, "smooth_attentions": True
                }
            )
            if current_loss < min_loss:
                final_seed = random_seed
                min_loss = current_loss
            print(f"  迭代 {iter_idx+1}/{boxConfig.latents_choose_iter} | 种子: {random_seed} | 损失: {current_loss:.4f}")
        print(f"最佳种子: {final_seed} | 最小损失: {min_loss:.4f}")

    # 6. 生成最终图片
    print(f"\n=== 生成图片 (种子: {final_seed}) ===")
    generation_result = pipe(
        base_prompt=base_prompt + quality_magic["en"],
        attention_store=attention_controller,
        negative_prompt=negative_prompt,
        width=img_width, height=img_height,
        num_inference_steps=hyper_params["num_inference_steps"],
        true_cfg_scale=hyper_params["true_cfg_scale"],
        generator=torch.Generator(device="cuda").manual_seed(final_seed),
        mask_inject_steps=hyper_params["mask_inject_steps"],
        attention_kwargs=attention_kwargs,
        gaussian_smoothing_kwargs={
            "sigma": 0.5, "kernel_size": 3, "smooth_attentions": True
        }
    )

    # 7. 保存结果
    raw_image_path = os.path.join(SAVE_BASE_PATH, "paper_demo_result.png")
    generation_result.images[0].save(raw_image_path)
    print(f"生成图片已保存至: {raw_image_path}")

    masked_image_path = os.path.join(SAVE_BASE_PATH, "paper_demo_with_masks.png")
    draw_masks_on_image(raw_image_path, regional_prompt_mask_pairs, masked_image_path)
    print(f"带掩码标注的图片已保存至: {masked_image_path}")

    # 8. 打印配置摘要
    print("\n" + "=" * 60)
    print("生成完成! 方法配置摘要:")
    print(f"  NQE-EAA: {'✓' if boxConfig.use_nqe_eaa else '✗'}")
    print(f"  动态软掩码: {'✓' if boxConfig.use_dynamic_mask else '✗'}")
    print(f"  LALO 损失: {boxConfig.lossType}")
    print(f"  最终种子: {final_seed}")
    print(f"  输出目录: {SAVE_BASE_PATH}")
    print("=" * 60)


# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        run_paper_demo()
    except Exception as e:
        print(f"\n生成过程出错: {e}")
        import traceback
        traceback.print_exc()
        raise
