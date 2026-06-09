"""
NQE-EAA: Noise Quality Evaluation via Early Attention Alignment
基于早期注意力对齐的噪声质量评估

Implements Algorithm 1 from the paper:
    基于早期交叉注意力分布的噪声质量评估方法，通过量化注意力在目标布局区域中的
    响应程度，衡量候选噪声对目标构图的布局对齐能力。

Reference: Section "潜在噪声筛选" (Potential Noise Filtering)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from config.boxLossConfig import boxConfig


def extract_attention_from_block(
    attention_store,
    block_indices: List[int],
    shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract text-to-image and image-to-text attention maps from specific DiT blocks
    and aggregate them into spatial maps.

    Args:
        attention_store: AttentionStore containing captured attention from all blocks
        block_indices: List of block indices to extract from
        shape: (H, W, scale_factor) — latent spatial dimensions

    Returns:
        txt2img_avg: Averaged text-to-image attention map [H_z, W_z]
        img2txt_avg: Averaged image-to-text attention map [H_z, W_z]
    """
    H, W, scale_factor = shape
    H_z, W_z = H // scale_factor, W // scale_factor

    attention_maps = attention_store.get_store_attention()

    txt2img_maps = []
    img2txt_maps = []

    # Aggregate txt-to-img attention across specified blocks
    if "txt-to-img" in attention_maps:
        for item in attention_maps["txt-to-img"]:
            # item shape: [B, heads, txt_seq, img_seq] — transpose to [B, heads, img_seq, txt_seq]
            item_t = item.transpose(-1, -2)
            # Reshape to spatial: [B, heads, H_z, W_z, txt_seq]
            cross_maps = item_t.reshape(
                item_t.shape[0], item_t.shape[1], H_z, W_z, item_t.shape[-1]
            )
            # Average over heads and text tokens -> [H_z, W_z]
            txt2img_maps.append(cross_maps.mean(dim=(0, 1, -1)))

    # Aggregate img-to-txt attention across specified blocks
    if "img-to-txt" in attention_maps:
        for item in attention_maps["img-to-txt"]:
            # item shape: [B, heads, img_seq, txt_seq]
            # Reshape to spatial: [B, heads, H_z, W_z, txt_seq]
            cross_maps = item.reshape(
                item.shape[0], item.shape[1], H_z, W_z, item.shape[-1]
            )
            # Average over heads and text tokens -> [H_z, W_z]
            img2txt_maps.append(cross_maps.mean(dim=(0, 1, -1)))

    txt2img_avg = torch.stack(txt2img_maps).mean(dim=0) if txt2img_maps else None
    img2txt_avg = torch.stack(img2txt_maps).mean(dim=0) if img2txt_maps else None

    return txt2img_avg, img2txt_avg


def compute_single_noise_score(
    attention_store,
    bboxes: List[List[int]],
    shape: Tuple[int, int, int],
    alpha: float = 0.5,
) -> float:
    """
    Compute the NQE-EAA score for a single candidate noise.

    Args:
        attention_store: AttentionStore with captured attention from one forward pass
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...] in pixel coordinates
        shape: (H, W, scale_factor) — image dimensions
        alpha: Balance coefficient between txt→img and img→txt attention

    Returns:
        s_k: Weighted noise quality score (higher = better alignment with layout)
    """
    H, W, scale_factor = shape
    H_z, W_z = H // scale_factor, W // scale_factor

    txt2img_avg, img2txt_avg = extract_attention_from_block(
        attention_store, [], shape
    )

    # If one direction is missing, use only the available one
    if txt2img_avg is None and img2txt_avg is None:
        return 0.0

    # Compute global mean thresholds
    if txt2img_avg is not None:
        mu_txt2img = txt2img_avg.mean().item()
    else:
        mu_txt2img = 0.0

    if img2txt_avg is not None:
        mu_img2txt = img2txt_avg.mean().item()
    else:
        mu_img2txt = 0.0

    total_area = H_z * W_z
    s_k = 0.0

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # Scale bbox from pixel space to latent space
        x1_z = max(round(x1 / scale_factor), 0)
        y1_z = max(round(y1 / scale_factor), 0)
        x2_z = min(round(x2 / scale_factor), W_z)
        y2_z = min(round(y2 / scale_factor), H_z)

        if x2_z <= x1_z or y2_z <= y1_z:
            continue

        mask_area = (x2_z - x1_z) * (y2_z - y1_z)
        if mask_area == 0:
            continue

        # Compute txt→img score: proportion of pixels inside bbox above global mean
        a_txt2img = 0.0
        if txt2img_avg is not None:
            inside_region = txt2img_avg[y1_z:y2_z, x1_z:x2_z]
            above_mean = (inside_region > mu_txt2img).float().sum().item()
            a_txt2img = above_mean / mask_area

        # Compute img→txt score: proportion of pixels inside bbox above global mean
        a_img2txt = 0.0
        if img2txt_avg is not None:
            inside_region = img2txt_avg[y1_z:y2_z, x1_z:x2_z]
            above_mean = (inside_region > mu_img2txt).float().sum().item()
            a_img2txt = above_mean / mask_area

        # Weighted fusion of bidirectional attention
        s_i = alpha * a_txt2img + (1.0 - alpha) * a_img2txt

        # Area-weighted contribution
        w_i = mask_area / total_area
        s_k += w_i * s_i

    return s_k


def compute_nqe_eaa_scores(
    pipe,
    candidate_noises: List[torch.Tensor],
    base_prompt: str,
    bboxes: List[List[int]],
    height: int,
    width: int,
    attention_store,
    gaussian_smoothing_kwargs: Optional[Dict] = None,
    alpha: float = 0.5,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 4,
    mask_inject_steps: int = 0,
    attention_kwargs: Optional[Dict] = None,
) -> List[float]:
    """
    Evaluate NQE-EAA scores for multiple candidate noise samples.

    Performs one early forward pass per candidate noise to extract attention maps
    from ALL DiT blocks, then computes the layout alignment score.

    Args:
        pipe: RegionalQwenImagePipeline instance
        candidate_noises: List of latent noise tensors [z_1, z_2, ..., z_K]
        base_prompt: Text prompt
        bboxes: Target bounding boxes in pixel coordinates
        height: Image height
        width: Image width
        attention_store: AttentionStore instance for capturing attention
        gaussian_smoothing_kwargs: Gaussian smoothing parameters
        alpha: Balance coefficient between txt→img and img→txt
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of inference steps
        mask_inject_steps: Mask injection steps (set to 0 for early evaluation)
        attention_kwargs: Regional attention kwargs

    Returns:
        scores: List of NQE-EAA scores for each candidate noise
    """
    scores = []
    shape = (height, width, pipe.vae_scale_factor * 2)
    device = pipe._execution_device

    # Prepare text embeddings once (shared across all candidates)
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt=base_prompt,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    # Prepare regional embeddings if attention_kwargs provided
    regional_embeds = None
    regional_embeds_mask = None
    regional_txt_seq_lens = None
    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None

    if attention_kwargs is not None and 'regional_prompts' in attention_kwargs:
        regional_inputs = []
        each_prompt_seq_len = []
        for regional_prompt, regional_mask in zip(
            attention_kwargs['regional_prompts'],
            attention_kwargs['regional_masks']
        ):
            r_embeds, r_mask = pipe.encode_prompt(
                prompt=regional_prompt,
                prompt_embeds=None,
                prompt_embeds_mask=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            regional_inputs.append((regional_mask, r_embeds, r_mask))
            each_prompt_seq_len.append(r_embeds.shape[1])

        if regional_inputs:
            conds = [ri[1] for ri in regional_inputs]
            cond_masks = [ri[2] for ri in regional_inputs]
            regional_embeds = torch.cat(conds, dim=1)
            regional_embeds_mask = torch.cat(cond_masks, dim=1)
            regional_txt_seq_lens = regional_embeds_mask.sum(dim=1).tolist()

    # Handle guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance = None

    img_shapes = [(1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)]

    # Use the very first timestep (t ≈ T, early denoising) for evaluation
    # This corresponds to the paper's "early attention" analysis
    sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    t_eval = sigmas[0]  # First timestep

    for k, z_k in enumerate(candidate_noises):
        # Reset attention store for this candidate
        attention_store.reset()

        latents = z_k.clone().to(device).requires_grad_(False)

        timestep = torch.full([1], t_eval, device=device, dtype=latents.dtype)

        # Single forward pass to extract attention from ALL blocks
        # Use base prompt (no regional mask injection at early stage)
        with torch.no_grad():
            chosen_embeds = regional_embeds if (mask_inject_steps > 0 and regional_embeds is not None) else prompt_embeds
            chosen_mask = regional_embeds_mask if (mask_inject_steps > 0 and regional_embeds_mask is not None) else prompt_embeds_mask
            chosen_txt_seq_lens = regional_txt_seq_lens if (mask_inject_steps > 0 and regional_txt_seq_lens is not None) else txt_seq_lens

            _ = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=chosen_mask,
                encoder_hidden_states=chosen_embeds,
                img_shapes=img_shapes,
                regional_txt_seq_lens=chosen_txt_seq_lens,
                attention_kwargs={"double_inject_blocks_interval": 1}
                if attention_kwargs is None else attention_kwargs,
                return_dict=False,
            )[0]

        # Compute NQE-EAA score
        score = compute_single_noise_score(
            attention_store=attention_store,
            bboxes=bboxes,
            shape=shape,
            alpha=alpha,
        )
        scores.append(score)

    return scores


def select_best_noise(
    pipe,
    candidate_noises: List[torch.Tensor],
    base_prompt: str,
    bboxes: List[List[int]],
    height: int,
    width: int,
    attention_store,
    alpha: float = 0.5,
    **kwargs,
) -> Tuple[torch.Tensor, int, float]:
    """
    Select the best initial noise from K candidates using NQE-EAA.

    Args:
        pipe: RegionalQwenImagePipeline instance
        candidate_noises: List of K candidate noise tensors
        base_prompt: Text prompt
        bboxes: Target bounding boxes
        height: Image height
        width: Image width
        attention_store: AttentionStore instance
        alpha: Balance coefficient

    Returns:
        best_noise: The noise tensor with highest NQE-EAA score
        best_idx: Index of the best noise
        best_score: NQE-EAA score of the best noise
    """
    scores = compute_nqe_eaa_scores(
        pipe=pipe,
        candidate_noises=candidate_noises,
        base_prompt=base_prompt,
        bboxes=bboxes,
        height=height,
        width=width,
        attention_store=attention_store,
        alpha=alpha,
        **kwargs,
    )

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_noise = candidate_noises[best_idx]

    return best_noise, best_idx, best_score
