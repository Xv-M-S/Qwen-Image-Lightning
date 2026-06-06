"""
Qwen-Image 区域控制图片生成脚本
核心功能：基于指定的区域掩码和提示词，生成带精准文本/元素控制的咖啡店场景图片
"""
import os
import random
import math
from typing import Tuple, Dict, List

import torch
from pyinstrument import Profiler
from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
)

# 本地模块导入（按功能分组）
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
    visualize_structured_boxes_with_text
)
from util.textComposition import visualize_structured_boxes_with_text
from config.boxLossConfig import boxConfig
from testCase import regional_prompt_mask_pairs2 as regional_prompt_mask_pairs

# ===================== 全局配置常量 =====================
ENABLE_PROFILER = False  # 是否启用性能分析
SAVE_BASE_PATH = "./runing_output_tempfile"  # 输出文件基础路径
MODEL_NAME = "Qwen/Qwen-Image"  # 基础模型名称
LORA_PATH = "Qwen-Image-Lightning/Qwen-Image-Lightning-4steps-V1.0.safetensors"  # LORA权重路径

# ===================== 工具函数 =====================
def count_model_parameters(model: torch.nn.Module) -> int:
    """
    计算模型可训练参数数量
    :param model: 待计算的PyTorch模型
    :return: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size_gb(model: torch.nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """
    计算模型显存占用（GB）
    :param model: PyTorch模型
    :param dtype: 模型数据类型（默认float32）
    :return: 模型占用显存大小（GB）
    """
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4 if dtype == torch.float32 else 2  # float32=4字节, float16/bfloat16=2字节
    total_gb = (total_params * bytes_per_param) / (1024 ** 3)
    return round(total_gb, 2)


# ===================== 模型加载相关 =====================
def load_qwen_image_model() -> Tuple[RegionalQwenImagePipeline, AttentionStore]:
    """
    加载Qwen-Image模型及LORA权重，配置调度器和注意力控制器
    :return: (模型管道, 注意力控制器)
    """
    # 设备和数据类型配置
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
        print(f"使用GPU加速，数据类型: {torch_dtype}")
    else:
        torch_dtype = torch.float32
        device = "cpu"
        print("GPU不可用，使用CPU运行（速度较慢）")

    # 加载基础模型
    pipe_cls = RegionalQwenImagePipeline
    controller = AttentionStore()

    if LORA_PATH is not None:
        # 加载带LORA的模型
        transformer = RegionalQwenImageTransformer.from_pretrained(
            MODEL_NAME,
            subfolder="transformer",
            torch_dtype=torch_dtype
        )

        # 配置调度器参数
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

        # 构建模型管道并加载LORA
        pipe = pipe_cls.from_pretrained(
            MODEL_NAME,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch_dtype
        )
        pipe.load_lora_weights(LORA_PATH)
        
        # 注册注意力控制
        register_attention_control(pipe, controller=controller)
    else:
        # 加载基础模型（无LORA）
        pipe = pipe_cls.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)

    # 移动模型到指定设备，启用梯度检查点节省显存
    pipe = pipe.to(device)
    pipe.transformer.enable_gradient_checkpointing()

    # 打印模型信息
    print(f"模型加载完成，可训练参数: {count_model_parameters(pipe.transformer):,}")
    print(f"模型显存占用: {calculate_model_size_gb(pipe.transformer, torch_dtype)} GB")

    return pipe, controller


# ===================== 生成参数配置 =====================
def get_image_resolution() -> Tuple[int, int]:
    """
    获取图片生成的分辨率（按宽高比）
    :return: (高度, 宽度)
    """
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
    
    # 更新全局配置
    boxConfig.H = height
    boxConfig.W = width
    
    print(f"图片分辨率配置完成: {width}x{height} (16:9)")
    return height, width


def prepare_regional_control_params(height: int, width: int) -> Tuple[List[str], List[torch.Tensor], List, torch.Tensor, Dict, List]:
    """
    准备区域控制相关参数（提示词、掩码、框体等）
    :param height: 图片高度
    :param width: 图片宽度
    :return: (区域提示词列表, 区域掩码列表, 区域框列表, 整体区域掩码, 区域配置字典, 子框列表)
    """
    # 获取区域配置并预处理
    regional_prompt_mask_pairs_processed = get_child_boxes(regional_prompt_mask_pairs, width, height)
    print(f"区域配置预处理完成，共{len(regional_prompt_mask_pairs_processed)}个区域")
    
    # 修复：调整visualize_structured_boxes_with_text的参数（移除关键字参数，使用位置参数）
    # 匹配原始代码的调用方式，避免参数名不匹配问题
    visualize_structured_boxes_with_text(
        width,  # 第一个参数：宽度（位置参数）
        height, # 第二个参数：高度（位置参数）
        regional_prompt_mask_pairs_processed,  # 第三个参数：区域配置
        os.path.join(SAVE_BASE_PATH, "output_structured_boxes_with_text.png")  # 第四个参数：输出路径
    )

    # 初始化变量
    regional_prompts = []
    regional_masks = []
    regional_boxes = []
    regional_child_boxes = []
    background_prompt = "A coffee shop entrance, "
    background_mask = torch.ones((height, width))

    # 遍历每个区域，构建掩码和提示词
    for region_idx, region in regional_prompt_mask_pairs_processed.items():
        # 提取区域信息
        region_desc = region['description']
        region_box = region['mask']
        child_boxes = region['child_boxes']
        
        # 构建区域掩码（0=背景，1=区域）
        x1, y1, x2, y2 = region_box
        region_mask = torch.zeros((height, width))
        region_mask[y1:y2, x1:x2] = 1.0
        
        # 更新背景掩码（扣除当前区域）
        background_mask -= region_mask
        
        # 收集区域参数
        regional_prompts.append(region_desc)
        regional_masks.append(region_mask)
        regional_boxes.append(region_box)
        regional_child_boxes.append(child_boxes)

    # 添加背景区域（如果有未覆盖的区域）
    whole_regional_mask = torch.ones((height, width)) - background_mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)
        print("添加背景区域提示词，覆盖未指定的图片区域")

    return (
        regional_prompts,
        regional_masks,
        regional_boxes,
        whole_regional_mask,
        regional_prompt_mask_pairs_processed,
        regional_child_boxes
    )


def prepare_base_prompt_params() -> Tuple[str, str, Dict[str, str]]:
    """
    准备基础提示词参数（主提示词、负提示词、魔法词）
    :return: (基础提示词, 负提示词, 画质魔法词字典)
    """
    # 核心生成提示词（最终生效版本）
    base_prompt = '''A coffee shop entrance features a chalkboard sign reading "咖啡店" , 
    and a neon light displaying "通义千问" . the text "π≈3.1415926" is written on the wall .'''
    
    # 负提示词（无特殊要求时设为空字符串）
    negative_prompt = " "
    
    # 画质增强魔法词（分中英文）
    quality_magic_words = {
        "en": "Ultra HD, 4K, cinematic composition.",
        "zh": "超清，4K，电影级构图"
    }

    print("基础提示词配置完成")
    return base_prompt, negative_prompt, quality_magic_words


# ===================== 核心生成逻辑 =====================
def run_image_generation():
    """
    执行完整的图片生成流程：参数配置 → 模型加载 → 损失优化 → 图片生成 → 结果保存
    """
    # 1. 创建输出目录
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)
    
    # 2. 区域控制超参数（集中管理，方便调参）
    hyper_params = {
        "mask_inject_steps": 0,               # 掩码注入步数（越大控制越强，推荐5-10）
        "double_inject_blocks_interval": 1,   # 双注入块间隔（1=最强控制）
        "base_ratio": 0.1,                    # 基础比率（越小控制越强）
        "num_inference_steps": 4,             # 推理步数
        "true_cfg_scale": 1.0                 # CFG缩放（1.0=禁用）
    }

    # 3. 加载模型和配置参数
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
        img_width,
        img_height,
        os.path.join(SAVE_BASE_PATH, "visual_layout.png")
    )

    # 5. 优化随机种子（基于损失最小化）
    final_seed = 68
    min_loss = float('inf')
    if boxConfig.latents_choose_iter > 0:
        print(f"\n开始种子优化，迭代次数: {boxConfig.latents_choose_iter}")
        for iter_idx in range(boxConfig.latents_choose_iter):
            random_seed = random.randint(0, 1000000)
            print(f"迭代 {iter_idx+1}/{boxConfig.latents_choose_iter} | 随机种子: {random_seed} | 当前最小损失: {min_loss:.4f}")
            
            # 计算当前种子的损失
            current_loss = pipe.multi_step_loss(
                base_prompt=base_prompt + quality_magic["en"],
                attention_store=attention_controller,
                negative_prompt=negative_prompt,
                width=img_width,
                height=img_height,
                num_inference_steps=hyper_params["num_inference_steps"],
                true_cfg_scale=hyper_params["true_cfg_scale"],
                generator=torch.Generator(device="cuda").manual_seed(random_seed),
                mask_inject_steps=hyper_params["mask_inject_steps"],
                attention_kwargs={
                    "regional_prompts": regional_prompts,
                    "regional_masks": regional_masks,
                    "regional_boxes": regional_boxes,
                    "regional_child_boxes": regional_child_boxes,
                    "double_inject_blocks_interval": hyper_params["double_inject_blocks_interval"],
                    "base_ratio": hyper_params["base_ratio"],
                    "whole_regional_mask": whole_regional_mask,
                    "enable_whole_regional_mask": False
                },
                gaussian_smoothing_kwargs={
                    "sigma": 0.5,
                    "kernel_size": 3,
                    "smooth_attentions": True
                }
            )
            
            # 更新最优种子
            if current_loss < min_loss:
                final_seed = random_seed
                min_loss = current_loss

    print(f"\n种子优化完成 | 最优种子: {final_seed} | 最小损失: {min_loss:.4f}")

    # 6. 生成最终图片
    print("\n开始生成图片...")
    generation_result = pipe(
        base_prompt=base_prompt + quality_magic["en"],
        attention_store=attention_controller,
        negative_prompt=negative_prompt,
        width=img_width,
        height=img_height,
        num_inference_steps=hyper_params["num_inference_steps"],
        true_cfg_scale=hyper_params["true_cfg_scale"],
        generator=torch.Generator(device="cuda").manual_seed(final_seed),
        mask_inject_steps=hyper_params["mask_inject_steps"],
        attention_kwargs={
            "regional_prompts": regional_prompts,
            "regional_masks": regional_masks,
            "regional_boxes": regional_boxes,
            "regional_child_boxes": regional_child_boxes,
            "double_inject_blocks_interval": hyper_params["double_inject_blocks_interval"],
            "base_ratio": hyper_params["base_ratio"],
            "whole_regional_mask": whole_regional_mask,
            "enable_whole_regional_mask": False
        },
        gaussian_smoothing_kwargs={
            "sigma": 0.5,
            "kernel_size": 3,
            "smooth_attentions": True
        }
    )

    # 7. 保存生成结果
    raw_image_path = os.path.join(SAVE_BASE_PATH, "example.png")
    generation_result.images[0].save(raw_image_path)
    print(f"原始图片已保存至: {raw_image_path}")

    # 8. 可视化掩码在图片上的位置
    masked_image_path = os.path.join(SAVE_BASE_PATH, boxConfig.save_name)
    draw_masks_on_image(
        raw_image_path,
        regional_prompt_mask_pairs,
        masked_image_path
    )
    print(f"带掩码标注的图片已保存至: {masked_image_path}")

    print("\n图片生成流程全部完成！")


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 启用性能分析（可选）
    profiler = None
    if ENABLE_PROFILER:
        profiler = Profiler()
        profiler.start()
        print("性能分析器已启动...")

    # 执行生成流程
    try:
        run_image_generation()
    except Exception as e:
        print(f"\n生成过程出错: {e}")
        raise

    # 停止并打印性能分析结果
    if ENABLE_PROFILER and profiler:
        profiler.stop()
        profiler.print()